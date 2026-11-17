"""Access-control leak test: adversarial queries as a restricted role, retrieval only.

Run:
  uv run python scripts/acl_leak_test.py --config rerank                       # role public
  uv run python scripts/acl_leak_test.py --config structure --schema acl_scratch

For every query of `eval/acl_leak.jsonl` (simulated access model, `kaprag.access.leak`)
the config's retriever stack (dense / hybrid arms, reranking) retrieves under the
role's access filter, exactly as the pipeline does; no generation, no LLM call. A
config's `query_filter` step (a Haiku call) is OFF here, so the test is deterministic
and free: the access condition is the only filter. Its `access` block, if any, is
ignored: the role is `--role`.

- Leak: an item of the full ranked list (`--k` items, default the whole reranked pool,
  i.e. `rerank.candidates`, else `retrieve_k`) whose stored `access_roles` does not
  intersect the role's effective roles. Target 0; exit code 1 when leaks > 0.
- Positive control: the same queries with the access filter OFF; the number of queries
  whose list holds chunks the role may not see. A control with 0 such queries proves
  nothing (e.g. roles not assigned yet): exit code 2.

Output: `runs/YYYY-MM-DD-acl-leak-<config>-<role>[-<schema>][-N]/` with `README.md`,
`results.jsonl` (per query: filtered and control lists with roles, leaks) and
`summary.json`. Needs the `ml` group, a free GPU and the docker compose Postgres.
"""

import argparse
import json
import sys
from datetime import date
from pathlib import Path

from dotenv import load_dotenv
from psycopg import sql

from kaprag.access.filter import access_filter
from kaprag.access.leak import DEFAULT_LEAK_QUERIES, check_query, load_leak_queries, summarize
from kaprag.access.roles import ROLES, effective_roles
from kaprag.corpus import REPO_ROOT
from kaprag.eval.config import load_config
from kaprag.eval.runner import DEFAULT_RUNS_DIR, make_run_dir
from kaprag.index.store import connect, table_name
from kaprag.rag.config import RagConfig, parse_rag_config
from kaprag.rag.retriever import open_retriever


def default_depth(config: RagConfig) -> int:
    return config.rerank.candidates if config.rerank is not None else config.retrieve_k


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv(REPO_ROOT / ".env", override=False)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="configs/<name>.yaml (factory rag)")
    parser.add_argument("--role", default="public", choices=ROLES)
    parser.add_argument("--schema", default=None, help="schema of the chunk tables")
    parser.add_argument("--k", type=int, default=None, help="ranked items checked per query")
    parser.add_argument("--queries", type=Path, default=DEFAULT_LEAK_QUERIES)
    args = parser.parse_args()

    eval_config = load_config(args.config)
    if eval_config.pipeline.factory != "rag":
        parser.error(f"config {args.config!r} is not a RAG pipeline config")
    config = parse_rag_config(eval_config.pipeline.params)
    k = args.k or default_depth(config)
    queries = load_leak_queries(args.queries)
    table = table_name(config.index)

    retriever = open_retriever(config, schema=args.schema)
    try:
        ranked = {}
        for q in queries:
            filtered = retriever.retrieve(q.query, k, where=access_filter(args.role))
            control = retriever.retrieve(q.query, k, where=None)
            ranked[q.id] = (filtered, control)
    finally:
        retriever.close()

    ids = sorted({r.item.chunk_id for f, c in ranked.values() for r in (*f, *c)})
    with connect(schema=args.schema) as conn:
        rows = conn.execute(
            sql.SQL("SELECT chunk_id, access_roles FROM {} WHERE chunk_id = ANY(%s)").format(
                sql.Identifier(table)
            ),
            (ids,),
        ).fetchall()
    roles_of = {chunk_id: roles for chunk_id, roles in rows}

    results, controls, records = [], [], []
    for q in queries:
        filtered, control = ranked[q.id]
        result = check_query(q, [r.item.chunk_id for r in filtered], roles_of, args.role)
        ctrl = check_query(q, [r.item.chunk_id for r in control], roles_of, args.role)
        results.append(result)
        controls.append(ctrl)
        text = {r.item.chunk_id: r.item.text for r in control}
        records.append(
            {
                "id": q.id,
                "attack": q.attack,
                "query": q.query,
                "target": q.target,
                "retrieved": [[cid, roles_of[cid]] for cid in result.retrieved],
                "leaked": result.leaked,
                "control_retrieved": [[cid, roles_of[cid]] for cid in ctrl.retrieved],
                "control_restricted": ctrl.leaked,
                "control_restricted_examples": [
                    {"chunk_id": cid, "roles": roles_of[cid], "text": (text[cid] or "")[:300]}
                    for cid in ctrl.leaked[:3]
                ],
            }
        )

    leaks = summarize(results)
    control = summarize(controls)
    by_attack = {
        kind: {
            "queries": sum(q.attack == kind for q in queries),
            "leaking": sum(r.attack == kind and bool(r.leaked) for r in results),
            "control_restricted": sum(c.attack == kind and bool(c.leaked) for c in controls),
        }
        for kind in ("direct", "injection")
    }
    summary = {
        "config": args.config,
        "index": config.index,
        "schema": args.schema,
        "role": args.role,
        "effective_roles": list(effective_roles(args.role)),
        "depth_k": k,
        "filtered_list_length_min": min(len(r.retrieved) for r in results),
        "query_filter_step": "off",
        "leaks": leaks,
        "positive_control_filter_off": {
            "queries_with_restricted": control["queries_with_leaks"],
            "restricted_items": control["leaked_items"],
            "restricted_chunks": len(control["leaked_chunk_ids"]),
            # the same, counting only the top retrieve_k / context_k of each list
            **{
                f"queries_with_restricted_in_top{n}": sum(
                    bool(set(c.retrieved[:n]) & set(c.leaked)) for c in controls
                )
                for n in (config.retrieve_k, config.context_k)
            },
        },
        "by_attack": by_attack,
    }
    split = "-".join(part for part in (args.config, args.role, args.schema) if part)
    run_dir = make_run_dir(DEFAULT_RUNS_DIR, "acl-leak", split, date.today())
    where = f"{config.index} index" + (f", schema {args.schema}" if args.schema else "")
    (run_dir / "README.md").write_text(
        f"Access-control leak test: {len(queries)} adversarial queries as role {args.role} "
        f"via config {args.config} ({where}), retrieval only, top {k} checked; "
        "positive control with the filter off.\n",
        encoding="utf-8",
    )
    with (run_dir / "results.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"run folder: {run_dir}")
    print(
        f"{args.config} ({config.index}) as {args.role}, top {k}: "
        f"{leaks['queries_with_leaks']}/{len(queries)} queries leak, "
        f"{leaks['leaked_items']} leaked items {leaks['leaked_chunk_ids'][:10]}"
    )
    pc = summary["positive_control_filter_off"]
    print(
        f"positive control (filter off): {control['queries_with_leaks']}/{len(queries)} queries "
        f"retrieve restricted chunks in the top {k} ({control['leaked_items']} items, "
        f"{len(control['leaked_chunk_ids'])} distinct); in the top {config.retrieve_k}: "
        f"{pc[f'queries_with_restricted_in_top{config.retrieve_k}']}, in the top "
        f"{config.context_k}: {pc[f'queries_with_restricted_in_top{config.context_k}']}"
    )
    for kind, n in by_attack.items():
        print(f"  {kind}: {n}")
    if leaks["queries_with_leaks"]:
        return 1
    if not control["queries_with_leaks"]:
        print("positive control found no restricted chunk: the test proves nothing")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
