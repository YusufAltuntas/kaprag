"""Draft the gold question set (eval/README.md §7) into a run folder and a questions file.

Every draft is `author: llm`, `verified: false`; only the user verifies, in the review app.

Run (full; resumable - re-running replays cached LLM calls and continues failed ones):
  uv run python scripts/draft_gold.py runs/2026-11-17-gold-draft --out eval/gold/questions.jsonl

An existing `--out` file is replaced only with `--overwrite`, and never when it holds a
verified or human-authored question (those come only from the review app).

Smoke (tiny scale, same code):
  uv run python scripts/draft_gold.py runs/2026-11-17-gold-draft-smoke \
      --out runs/2026-11-17-gold-draft-smoke/questions.jsonl \
      --docs akbank-2024,akbank-2025,isbank-2025,garanti-2025 --table-pages 1 --text-pages 1 \
      --metrics net_profit,total_assets,branches --abstain-per-call 1 --ambiguous-per-kind 1

Needs the index tables (`--index-variant`, all 12 documents present), `DATABASE_URL`,
the `ml` group and a GPU for query embeddings, and `LLM_BACKEND` (default claude_code).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from kaprag.corpus import REPO_ROOT, read_manifest
from kaprag.gold.assemble import DEFAULT_QUOTAS
from kaprag.gold.guard import overwrite_refusal
from kaprag.gold.metrics import METRICS, METRICS_BY_ID
from kaprag.gold.pipeline import Pipeline, RunConfig, trace_summary
from kaprag.gold.search import open_searcher, table_complete
from kaprag.gold.stats import distribution_report
from kaprag.index.store import VARIANTS
from kaprag.llm import create_client


def _counts(text: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for part in filter(None, (p.strip() for p in text.split(","))):
        name, _, n = part.partition("=")
        out[name.strip()] = int(n)
    return out


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv(REPO_ROOT / ".env", override=False)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path, help="run folder, e.g. runs/2026-11-17-gold-draft")
    parser.add_argument("--out", type=Path, required=True, help="questions JSONL to write")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing --out file (never one with verified or human questions)",
    )
    parser.add_argument("--docs", default="", help="comma-separated doc ids (default: all)")
    parser.add_argument("--table-pages", type=int, default=4, help="table pages per document")
    parser.add_argument("--text-pages", type=int, default=4, help="text pages per document")
    parser.add_argument("--table-requests", default="tablo=2,hesaplama=1")
    parser.add_argument("--text-requests", default="olgusal=2,sentez=1")
    parser.add_argument("--metrics", default=",".join(m.id for m in METRICS))
    parser.add_argument("--abstain-per-call", type=int, default=4, help="per kind x focus call")
    parser.add_argument("--abstain-batch", type=int, default=4)
    parser.add_argument("--ambiguous-per-kind", type=int, default=9)
    parser.add_argument("--quotas", default=",".join(f"{t}={n}" for t, n in DEFAULT_QUOTAS.items()))
    parser.add_argument("--pilot", type=int, default=30)
    parser.add_argument("--index-variant", default="structure", choices=VARIANTS)
    parser.add_argument("--device", default="cuda", help="query embedding device")
    parser.add_argument("--no-evidence", action="store_true", help="skip evidence completion")
    parser.add_argument("--evidence-model", default="claude-haiku-5-5")
    args = parser.parse_args()

    if refusal := overwrite_refusal(args.out, overwrite=args.overwrite):
        print(refusal, file=sys.stderr)
        return 2
    backend = os.environ.get("LLM_BACKEND", "claude_code")
    if backend == "claude_code" and os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is set; unset it for the claude_code backend", file=sys.stderr)
        return 2
    manifest = read_manifest()
    wanted = [d for d in args.docs.split(",") if d]
    docs = [d for d in manifest if not wanted or d.doc_id in wanted]
    if unknown := set(wanted) - {d.doc_id for d in docs}:
        parser.error(f"unknown doc ids: {sorted(unknown)}")
    counts = table_complete(args.index_variant, [d.doc_id for d in manifest])
    if missing := [d for d, n in counts.items() if n == 0]:
        print(f"index table chunks_{args.index_variant} lacks {missing}", file=sys.stderr)
        return 2

    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    readme = run_dir / "README.md"
    if not readme.exists():
        readme.write_text(
            "Gold set drafting run (T6): page drafts + checks, metric probes, derived, "
            f"cevapsiz and belirsiz drafts; args: {' '.join(sys.argv[1:])}\n",
            encoding="utf-8",
        )
    cfg = RunConfig(
        run_dir=run_dir,
        out=args.out.resolve(),
        docs=docs,
        table_pages=args.table_pages,
        text_pages=args.text_pages,
        table_requests=_counts(args.table_requests),
        text_requests=_counts(args.text_requests),
        metrics=[METRICS_BY_ID[m] for m in args.metrics.split(",") if m],
        abstain_per_call=args.abstain_per_call,
        abstain_batch=args.abstain_batch,
        ambiguous_per_kind=args.ambiguous_per_kind,
        quotas=_counts(args.quotas),
        pilot=args.pilot,
        evidence=not args.no_evidence,
        evidence_model=args.evidence_model,
    )
    trace_path = run_dir / "llm-trace.jsonl"
    client = create_client(trace_path=trace_path)
    start = time.perf_counter()
    summary = asyncio.run(
        Pipeline(cfg, client, lambda: open_searcher(args.index_variant, args.device)).run()
    )
    elapsed = time.perf_counter() - start
    summary["wall_time_s"] = round(elapsed, 1)
    summary["llm"] = trace_summary(trace_path)
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    from kaprag.eval.schema import load_questions

    print()
    print(distribution_report(load_questions(cfg.out)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
