"""Map a finished run folder (`config.yaml`, `traces.jsonl`, `results.jsonl`) to Langfuse items.

Pure functions, no Langfuse SDK: the items are plain dataclasses that
`kaprag.observability.langfuse_sink` turns into OpenTelemetry spans and scores.
JSONL stays the source of truth; nothing is recomputed (cost is the recorded
API-equivalent `cost_usd`, timestamps are the recorded ones).

Per question (one `traces.jsonl` record) one trace:

- root span `<qid>`: input = question, output = structured answer, ERROR when the
  question failed. Trace attributes: name = config name, session = run folder,
  tags, metadata = run config + question fields (the sink also copies them, minus
  large metadata values, to every span).
- `retrieval` (retriever): the retrieved chunks with scores; `context_chunk_ids` and
  `retrieval_trace` from `pipeline_debug` as metadata. Retrieval time is not recorded,
  so its bounds are inferred (between the last pipeline call before the first
  `generate` call and that call; the whole question otherwise).
- `router` (agent), only when `pipeline_debug.router` exists: child spans
  `classify`, `decompose`, `fallback`, `lookup:<id>`, `compute` (tool),
  `synthesize`, each holding the LLM calls of its step. Spans without calls of their
  own (compute, fallback) are instants after the preceding step.
- one generation per LLM call: model, usage, recorded cost, cache flag, latency
  (start = `ts - latency_ms`, end = `ts`). Judge calls hang under a `judge`
  evaluator span, pipeline calls under their router step or the root.
- scores from the question's `results.jsonl` metrics (names as in `summary.json`).

Ids are deterministic so a re-export can recognise what is already there: trace id
= sha256("<run folder>/<qid>")[:16 bytes] as hex (the scheme of
`Langfuse.create_trace_id(seed=...)`), observation id = sha256 of
"<run folder>/<qid>/<key>"[:8 bytes] as hex, score id = "<trace id>-<score name>".
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import yaml

from kaprag.eval.summary import METRICS

ROOT_KEY = "root"


def trace_id_for(run_key: str, qid: str) -> str:
    """32 lowercase hex chars, deterministic in (run folder, question id)."""
    return hashlib.sha256(f"{run_key}/{qid}".encode()).digest()[:16].hex()


def observation_id_for(run_key: str, qid: str, key: str) -> str:
    """16 lowercase hex chars (an OpenTelemetry span id), deterministic in its key."""
    return hashlib.sha256(f"{run_key}/{qid}/{key}".encode()).digest()[:8].hex()


@dataclass
class ObservationItem:
    id: str
    parent_id: str | None
    type: str  # span | generation | retriever | agent | tool | evaluator
    name: str
    start: datetime
    end: datetime
    input: Any = None
    output: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
    level: str = "DEFAULT"  # DEFAULT | ERROR
    status_message: str | None = None
    model: str | None = None
    usage_details: dict[str, int] | None = None
    cost_details: dict[str, float] | None = None


@dataclass
class TraceItem:
    id: str
    qid: str
    name: str
    session_id: str
    tags: list[str]
    metadata: dict[str, Any]
    observations: list[ObservationItem]  # the root first

    @property
    def root(self) -> ObservationItem:
        return self.observations[0]


@dataclass
class ScoreItem:
    id: str
    trace_id: str
    name: str
    value: float
    data_type: str  # NUMERIC | BOOLEAN
    timestamp: datetime
    comment: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class RunExport:
    run_key: str
    traces: list[TraceItem]
    scores: list[ScoreItem]
    errors: list[str]

    def counts(self) -> dict[str, int]:
        observations = [o for t in self.traces for o in t.observations]
        return {
            "traces": len(self.traces),
            "observations": len(observations),
            "generations": sum(o.type == "generation" for o in observations),
            "scores": len(self.scores),
            "errors": len(self.errors),
        }


class RunFolderError(Exception):
    """The folder is not a run folder (no readable `config.yaml` / `traces.jsonl`)."""


def load_run(run_dir: Path, limit: int | None = None) -> RunExport:
    """Build every export item of a run. Bad lines are collected in `errors`, not raised."""
    run_dir = Path(run_dir)
    config_path = run_dir / "config.yaml"
    traces_path = run_dir / "traces.jsonl"
    for path in (config_path, traces_path):
        if not path.is_file():
            raise RunFolderError(f"not a run folder: {path} not found")
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise RunFolderError(f"{config_path}: {exc}") from exc
    run_key = run_dir.resolve().name
    errors: list[str] = []
    results = _read_results(run_dir / "results.jsonl", errors)
    run_metadata = _run_metadata(config, run_key)

    traces: list[TraceItem] = []
    scores: list[ScoreItem] = []
    with traces_path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            if limit is not None and len(traces) >= limit:
                break
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                result = results.get(record.get("qid")) if isinstance(record, dict) else None
                trace = build_trace(record, run_key, run_metadata, result)
                trace_scores = build_scores(record, trace.id, result)
            except (ValueError, KeyError, TypeError, AttributeError) as exc:
                errors.append(f"{traces_path.name}:{lineno}: {type(exc).__name__}: {exc}")
                continue
            traces.append(trace)
            scores.extend(trace_scores)
    return RunExport(run_key=run_key, traces=traces, scores=scores, errors=errors)


def _read_results(path: Path, errors: list[str]) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        errors.append(f"{path.name}: not found; no scores exported")
        return {}
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                rows[row["id"]] = row
            except (ValueError, KeyError, TypeError) as exc:
                errors.append(f"{path.name}:{lineno}: {type(exc).__name__}: {exc}")
    return rows


def _run_metadata(config: dict[str, Any], run_key: str) -> dict[str, Any]:
    run = config.get("run") or {}
    pipeline = config.get("pipeline") or {}
    judge = config.get("judge") or {}
    return {
        "run_folder": run_key,
        "run_id": run.get("run_id"),
        "config_name": run.get("config_name"),
        "split": run.get("split"),
        "limit": run.get("limit"),
        "n_questions": run.get("n_questions"),
        "git_commit": run.get("git_commit"),
        "questions_path": run.get("questions_path"),
        "questions_sha256": run.get("questions_sha256"),
        "run_started_at": run.get("started_at"),
        "description": config.get("description"),
        "pipeline_factory": pipeline.get("factory"),
        "pipeline_params": pipeline.get("params"),
        "judge_kind": judge.get("kind"),
        "judge_model": judge.get("model"),
    }


def parse_ts(value: str) -> datetime:
    """ISO timestamp from the JSONL; a value without offset is UTC."""
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


class _Ids:
    def __init__(self, run_key: str, qid: str) -> None:
        self.run_key, self.qid = run_key, qid

    def __call__(self, key: str) -> str:
        return observation_id_for(self.run_key, self.qid, key)


def build_trace(
    record: dict[str, Any],
    run_key: str,
    run_metadata: dict[str, Any],
    result: dict[str, Any] | None = None,
) -> TraceItem:
    """One `traces.jsonl` record -> one trace with all its observations."""
    qid = record["qid"]
    if not isinstance(qid, str) or not qid:
        raise ValueError(f"bad qid {qid!r}")
    ids = _Ids(run_key, qid)
    started_at, ended_at = parse_ts(record["started_at"]), parse_ts(record["ended_at"])
    status = record["status"]
    debug = record.get("pipeline_debug") or {}
    router = debug.get("router")
    judge = record.get("judge") or {}
    root_id = ids(ROOT_KEY)

    generations = [_generation(call, i, ids) for i, call in enumerate(record["llm_calls"])]
    pipeline_gens = [g for g in generations if g.metadata["role"] != "judge"]
    judge_gens = [g for g in generations if g.metadata["role"] == "judge"]

    children: list[ObservationItem] = []
    if status != "error":
        children.append(_retrieval(record, debug, pipeline_gens, root_id, ids))
    placed: set[str] = set()
    if router:
        children.extend(_router(router, debug, pipeline_gens, started_at, root_id, ids, placed))
    for gen in pipeline_gens:
        if gen.id not in placed:
            gen.parent_id = root_id
            children.append(gen)
    if judge_gens or judge.get("task"):
        children.extend(_judge(judge, judge_gens, ended_at, root_id, ids))

    starts = [started_at, *(o.start for o in children)]
    ends = [ended_at, *(o.end for o in children)]
    root = ObservationItem(
        id=root_id,
        parent_id=None,
        type="span",
        name=qid,
        start=min(starts),
        end=max(ends),
        input={"question": record["question"]},
        output=record.get("answer"),
        metadata={"pipeline_latency_ms": record.get("latency_ms")},
        level="ERROR" if status == "error" else "DEFAULT",
        status_message=record.get("error"),
    )

    route = debug.get("route")
    tags = [
        f"run:{run_key}",
        f"config:{run_metadata.get('config_name')}",
        f"split:{run_metadata.get('split')}",
        f"type:{record['type']}",
        f"status:{status}",
        *([f"route:{route}"] if route else []),
    ]
    metadata = {
        **run_metadata,
        "qid": qid,
        "question_type": record["type"],
        "status": status,
        "error": record.get("error"),
        "latency_ms": record.get("latency_ms"),
        "verified": None if result is None else result.get("verified"),
        "judge_task": judge.get("task"),
        "judge_error": judge.get("error"),
    }
    return TraceItem(
        id=trace_id_for(run_key, qid),
        qid=qid,
        name=str(run_metadata.get("config_name") or run_key),
        session_id=run_key,
        tags=tags,
        metadata=metadata,
        observations=[root, *children],
    )


def _generation(call: dict[str, Any], index: int, ids: _Ids) -> ObservationItem:
    tags = call.get("tags") or {}
    role = call.get("role") or ("judge" if tags.get("role") == "judge" else "pipeline")
    step = tags.get("step")
    if role == "judge":
        name = f"judge:{tags['judge']}" if tags.get("judge") else "judge"
    else:
        name = step or "llm"
        if tags.get("lookup"):
            name = f"{name}:{tags['lookup']}"
    end = parse_ts(call["ts"])
    latency_ms = float(call.get("latency_ms") or 0.0)
    usage = {
        "input": call.get("input_tokens") or 0,
        "output": call.get("output_tokens") or 0,
        "cache_read_input_tokens": call.get("cache_read_input_tokens") or 0,
        "cache_creation_input_tokens": call.get("cache_creation_input_tokens") or 0,
    }
    error = call.get("error")
    return ObservationItem(
        id=ids(f"llm/{index}"),
        parent_id=None,  # set when placed
        type="generation",
        name=name,
        start=end - timedelta(milliseconds=latency_ms),
        end=end,
        input={"prompt_hash": call.get("prompt_hash")},
        metadata={
            "role": role,
            "step": step,
            "lookup": tags.get("lookup"),
            "judge": tags.get("judge"),
            "cached": bool(call.get("cached")),
            "latency_ms": call.get("latency_ms"),
            "source_latency_ms": call.get("source_latency_ms"),
            "attempts": call.get("attempts"),
            "backend": call.get("backend"),
            "prompt_hash": call.get("prompt_hash"),
            "cost_usd": call.get("cost_usd"),
            "tags": tags,
        },
        level="ERROR" if error else "DEFAULT",
        status_message=error,
        model=call.get("model"),
        usage_details=usage,
        cost_details={"total": float(call.get("cost_usd") or 0.0)},
    )


def _retrieval(
    record: dict[str, Any],
    debug: dict[str, Any],
    pipeline_gens: list[ObservationItem],
    root_id: str,
    ids: _Ids,
) -> ObservationItem:
    started_at, ended_at = parse_ts(record["started_at"]), parse_ts(record["ended_at"])
    generate = [g for g in pipeline_gens if g.metadata["step"] == "generate"]
    if generate:
        end = min(g.start for g in generate)
        before = [g.end for g in pipeline_gens if g.end <= end]
        start = max([started_at, *before])
    else:
        start, end = started_at, ended_at
    return ObservationItem(
        id=ids("retrieval"),
        parent_id=root_id,
        type="retriever",
        name="retrieval",
        start=start,
        end=max(start, end),
        input={"question": record["question"]},
        output=record.get("retrieval") or [],
        metadata={
            "timing": "inferred",
            "context_chunk_ids": debug.get("context_chunk_ids"),
            "retrieval_trace": debug.get("retrieval_trace"),
        },
    )


def _step_gens(
    gens: list[ObservationItem], placed: set[str], step: str, lookup: str | None = None
) -> list[ObservationItem]:
    found = [
        g
        for g in gens
        if g.id not in placed
        and g.metadata["step"] == step
        and (lookup is None or g.metadata["lookup"] == lookup)
    ]
    placed.update(g.id for g in found)
    return found


def _router(
    router: dict[str, Any],
    debug: dict[str, Any],
    pipeline_gens: list[ObservationItem],
    started_at: datetime,
    root_id: str,
    ids: _Ids,
    placed: set[str],
) -> list[ObservationItem]:
    """The router span, one span per recorded step, the step's LLM calls under it."""
    router_id = ids("router")
    out: list[ObservationItem] = []
    steps: list[ObservationItem] = []
    cursor = started_at

    def add_step(
        key: str,
        name: str,
        kind: str,
        gens: list[ObservationItem],
        *,
        input: Any = None,
        output: Any = None,
        metadata: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        nonlocal cursor
        step_id = ids(f"router/{key}")
        start = min((g.start for g in gens), default=cursor)
        end = max((g.end for g in gens), default=cursor)
        steps.append(
            ObservationItem(
                id=step_id,
                parent_id=router_id,
                type=kind,
                name=name,
                start=start,
                end=end,
                input=input,
                output=output,
                metadata=metadata or {},
                level="ERROR" if error else "DEFAULT",
                status_message=error,
            )
        )
        for g in gens:
            g.parent_id = step_id
        out.extend(gens)
        cursor = max(cursor, end)

    classify = router.get("classify")
    if classify is not None:
        add_step(
            "classify", "classify", "span", _step_gens(pipeline_gens, placed, "classify"),
            output=classify, error=classify.get("error"),
        )  # fmt: skip
    decompose = router.get("decompose")
    if decompose is not None:
        add_step(
            "decompose", "decompose", "span", _step_gens(pipeline_gens, placed, "decompose"),
            output=decompose.get("plan"),
            metadata={"year_rule_fixes": decompose.get("year_rule_fixes")},
            error=decompose.get("error"),
        )  # fmt: skip
    if router.get("fallback") is not None:
        add_step("fallback", "fallback", "span", [], output=router["fallback"])
    plan = (decompose or {}).get("plan") or {}
    planned = {lk.get("id"): lk for lk in plan.get("lookups") or [] if isinstance(lk, dict)}
    for lookup in router.get("lookups") or []:
        lid = lookup["id"]
        spec = planned.get(lid, {})
        add_step(
            f"lookup/{lid}", f"lookup:{lid}", "chain",
            _step_gens(pipeline_gens, placed, "sub_answer", lid),
            input={"sub_question": spec.get("sub_question"), "filter": lookup.get("filter")},
            output=lookup.get("answer"),
            metadata={k: v for k, v in lookup.items() if k not in ("id", "answer", "filter")},
            error=lookup.get("error"),
        )  # fmt: skip
    compute = router.get("compute")
    if compute is not None:
        error = compute.get("error") if isinstance(compute, dict) else None
        add_step("compute", "compute", "tool", [], output=compute, error=error)
    synthesize = router.get("synthesize")
    if synthesize is not None:
        error = synthesize.get("error") if isinstance(synthesize, dict) else None
        add_step(
            "synthesize", "synthesize", "span", _step_gens(pipeline_gens, placed, "synthesize"),
            output=synthesize, error=error,
        )  # fmt: skip

    router_span = ObservationItem(
        id=router_id,
        parent_id=root_id,
        type="agent",
        name="router",
        start=min((s.start for s in steps), default=started_at),
        end=max((s.end for s in steps), default=started_at),
        output={"route": debug.get("route")},
    )
    return [router_span, *steps, *out]


def _judge(
    judge: dict[str, Any],
    judge_gens: list[ObservationItem],
    ended_at: datetime,
    root_id: str,
    ids: _Ids,
) -> list[ObservationItem]:
    judge_id = ids("judge")
    for g in judge_gens:
        g.parent_id = judge_id
    error = judge.get("error")
    span = ObservationItem(
        id=judge_id,
        parent_id=root_id,
        type="evaluator",
        name="judge",
        start=min((g.start for g in judge_gens), default=ended_at),
        end=max((g.end for g in judge_gens), default=ended_at),
        input={"task": judge.get("task")},
        output=judge.get("verdict"),
        level="ERROR" if error else "DEFAULT",
        status_message=error,
    )
    return [span, *judge_gens]


def build_scores(
    record: dict[str, Any], trace_id: str, result: dict[str, Any] | None
) -> list[ScoreItem]:
    """The question's `results.jsonl` metrics as scores; null metrics are skipped.

    Scores are timestamped with the question's recorded end, so a re-export sends the
    same (id, name, date) and Langfuse overwrites instead of duplicating.
    """
    if result is None:
        return []
    metrics = result["metrics"]
    verdict = result.get("judge_verdict") or {}
    comments = {
        "numeric_correct": metrics.get("numeric_reason"),
        "text_correct": verdict.get("reason"),
        "key_point_coverage": verdict.get("reason"),
    }
    timestamp = parse_ts(record["ended_at"])
    scores = []
    for name, metric_field in METRICS.items():
        value = metrics.get(metric_field)
        if value is None:
            continue
        if isinstance(value, bool):
            numeric, data_type = (1.0 if value else 0.0), "BOOLEAN"
        elif isinstance(value, int | float):
            numeric, data_type = float(value), "NUMERIC"
        else:
            raise TypeError(f"metric {metric_field!r} is not numeric: {value!r}")
        scores.append(
            ScoreItem(
                id=f"{trace_id}-{name}",
                trace_id=trace_id,
                name=name,
                value=numeric,
                data_type=data_type,
                timestamp=timestamp,
                comment=comments.get(metric_field),
                metadata={"field": metric_field, "content_metric": metrics.get("content_metric")},
            )
        )
    return scores
