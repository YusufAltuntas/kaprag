"""Eval runner: answer every selected gold question with a pipeline, score, write a run folder.

Run folder `runs/YYYY-MM-DD-<topic>-<split>[-N]/` (never overwritten; a numeric suffix
is added when the name is taken):

- `README.md`: one line, the config's `description`.
- `config.yaml`: the resolved config plus `run` metadata (split, question file and its
  sha256, git commit, start time, limit).
- `results.jsonl`: one row per question (see `_result_row`).
- `summary.json`: `run` metadata, `overall` / `verified` / `by_type` groups (see
  `kaprag.eval.summary`), `judge` and `unattributed_llm_calls`.
- `traces.jsonl`: one self-contained record per question (see `_trace_record`).

LLM calls are attributed to questions through their trace tags: the runner hands the
pipeline `tags={"run", "qid", "role": "pipeline"}` and the judge tags its calls
`role: "judge"`. Any call with a `qid` and a role other than `judge` counts as the
pipeline's. The raw `kaprag.llm` trace of the run's client is written to a temporary
file and folded into `traces.jsonl`.

One question failing (any exception from the pipeline) is recorded as `status:
"error"` and the run continues. A judge failure leaves the question `ok` with
`judge_error` set and the judged metric null.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import subprocess
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import yaml

from kaprag.eval.config import EvalConfig
from kaprag.eval.judge import FakeJudge, Judge, LLMJudge, key_point_coverage
from kaprag.eval.metrics import needs_judge, score_question
from kaprag.eval.pipeline import PipelineContext, PipelineOutput, resolve_factory
from kaprag.eval.schema import Question
from kaprag.eval.summary import summarize
from kaprag.llm import LLMClient, create_client

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUNS_DIR = REPO_ROOT / "runs"
SPLITS = ("dev", "test", "pilot")

LLMFactory = Callable[[Path], LLMClient]  # trace path -> client


def select_questions(
    questions: Sequence[Question], split: str, limit: int | None = None
) -> list[Question]:
    """`pilot` selects `pilot: true`; `dev` / `test` select by `split`. File order kept."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
    if split == "pilot":
        selected = [q for q in questions if q.pilot]
    else:
        selected = [q for q in questions if q.split == split]
    return selected[:limit] if limit is not None else selected


def make_run_dir(runs_dir: Path, topic: str, split: str, day: date) -> Path:
    """Create a fresh `YYYY-MM-DD-<topic>-<split>` folder, suffixing `-2`, `-3`... if taken."""
    base = f"{day.isoformat()}-{topic}-{split}"
    runs_dir.mkdir(parents=True, exist_ok=True)
    n = 1
    while True:
        path = runs_dir / (base if n == 1 else f"{base}-{n}")
        try:
            path.mkdir()
            return path
        except FileExistsError:
            n += 1


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_commit() -> str | None:
    """HEAD commit of the repo, with `-dirty` when the work tree has changes."""
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return head + ("-dirty" if dirty else "")


@dataclass
class _QuestionRun:
    question: Question
    output: PipelineOutput | None
    status: str
    error: str | None
    started_at: str
    ended_at: str
    latency_ms: float
    judge_task: str | None
    judge_verdict: dict[str, Any] | None
    judge_error: str | None
    metrics: dict[str, Any]


@dataclass
class RunResult:
    run_dir: Path
    summary: dict[str, Any]


class _LazyLLM:
    """Creates the run's LLM client on first use, tracing to `trace_path`."""

    def __init__(self, factory: LLMFactory, trace_path: Path) -> None:
        self._factory = factory
        self.trace_path = trace_path
        self.client: LLMClient | None = None

    def __call__(self) -> LLMClient:
        if self.client is None:
            self.client = self._factory(self.trace_path)
        return self.client

    def rows(self) -> list[dict[str, Any]]:
        if self.client is None or not self.trace_path.exists():
            return []
        lines = self.trace_path.read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines if line.strip()]


def _default_llm_factory(trace_path: Path) -> LLMClient:
    return create_client(trace_path=trace_path)


def _now() -> str:
    return datetime.now(UTC).isoformat()


async def run_eval_async(
    config: EvalConfig,
    *,
    config_name: str,
    questions: Sequence[Question],
    questions_path: Path,
    split: str,
    runs_dir: Path = DEFAULT_RUNS_DIR,
    limit: int | None = None,
    llm_factory: LLMFactory = _default_llm_factory,
    today: date | None = None,
) -> RunResult:
    """Run one eval and write its folder. Raises only if the run cannot be completed."""
    selected = select_questions(questions, split, limit)
    if not selected:
        raise ValueError(f"no questions selected for split {split!r}")

    with tempfile.TemporaryDirectory(prefix="kaprag-eval-") as tmp:
        llm = _LazyLLM(llm_factory, Path(tmp) / "llm-trace.jsonl")
        factory = resolve_factory(config.pipeline.factory)
        pipeline = factory(
            PipelineContext(
                params=dict(config.pipeline.params), llm=llm, gold={q.id: q for q in questions}
            )
        )

        run_dir = make_run_dir(runs_dir, config.topic, split, today or date.today())
        run_meta = {
            "run_id": run_dir.name,
            "config_name": config_name,
            "split": split,
            "limit": limit,
            "n_questions": len(selected),
            "questions_path": _display_path(questions_path),
            "questions_sha256": sha256_file(questions_path),
            "git_commit": git_commit(),
            "started_at": _now(),
        }
        (run_dir / "README.md").write_text(config.description + "\n", encoding="utf-8")
        resolved = {**config.model_dump(mode="json"), "run": run_meta}
        (run_dir / "config.yaml").write_text(
            yaml.safe_dump(resolved, allow_unicode=True, sort_keys=False, width=10**6),
            encoding="utf-8",
        )

        judge_holder: list[Judge] = []

        def get_judge() -> Judge:
            if not judge_holder:
                if config.judge.kind == "fake":
                    judge_holder.append(FakeJudge())
                else:
                    judge_holder.append(LLMJudge(llm(), config.judge.model))
            return judge_holder[0]

        semaphore = asyncio.Semaphore(config.concurrency)
        wall_start = time.perf_counter()
        runs = await asyncio.gather(
            *(_run_question(q, pipeline, get_judge, semaphore, run_dir.name) for q in selected)
        )
        if hasattr(pipeline, "aclose"):
            try:
                await pipeline.aclose()
            except Exception:  # results are complete; a failed cleanup must not lose them
                log.exception("pipeline aclose failed")
        trace_rows = llm.rows()

    by_qid: dict[str, list[dict[str, Any]]] = {}
    unattributed = []
    for row in trace_rows:
        qid = row.get("tags", {}).get("qid")
        if qid is None:
            unattributed.append(row)
        else:
            by_qid.setdefault(qid, []).append(row)

    results, traces = [], []
    for run in runs:
        calls = sorted(by_qid.get(run.question.id, []), key=lambda r: r.get("ts", ""))
        results.append(_result_row(run, calls))
        traces.append(_trace_record(run, calls, run_dir.name))

    run_meta["finished_at"] = _now()
    run_meta["wall_time_s"] = round(time.perf_counter() - wall_start, 3)
    summary = {
        "run": {
            **run_meta,
            "description": config.description,
            "pipeline": config.pipeline.model_dump(mode="json"),
        },
        **summarize(results),
        "judge": {
            "kind": config.judge.kind,
            "model": config.judge.model if config.judge.kind == "llm" else None,
            "judged_questions": sum(r["judge_task"] is not None for r in results),
            "llm_calls": sum(r["judge_llm_calls"] for r in results),
            "errors": sum(r["judge_error"] is not None for r in results),
            "cost_usd_total": round(float(sum(r["judge_cost_usd"] for r in results)), 6),
        },
        "unattributed_llm_calls": {
            "n": len(unattributed),
            "cost_usd_total": round(float(sum(r.get("cost_usd") or 0 for r in unattributed)), 6),
        },
    }
    _write_jsonl(run_dir / "results.jsonl", results)
    _write_jsonl(run_dir / "traces.jsonl", traces)
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return RunResult(run_dir=run_dir, summary=summary)


def run_eval(config: EvalConfig, **kwargs: Any) -> RunResult:
    """Blocking wrapper around `run_eval_async`."""
    return asyncio.run(run_eval_async(config, **kwargs))


async def _run_question(
    question: Question,
    pipeline: Any,
    get_judge: Callable[[], Judge],
    semaphore: asyncio.Semaphore,
    run_id: str,
) -> _QuestionRun:
    async with semaphore:
        started_at = _now()
        start = time.perf_counter()
        tags = {"run": run_id, "qid": question.id, "role": "pipeline"}
        output: PipelineOutput | None = None
        error = None
        try:
            raw = await pipeline.run(question.question, tags=dict(tags))
            output = PipelineOutput.model_validate(raw)
        except Exception as exc:  # any pipeline failure is recorded; the run goes on
            error = f"{type(exc).__name__}: {exc}"
        latency_ms = (time.perf_counter() - start) * 1000
        ended_at = _now()

        task = needs_judge(question, output)
        verdict = judge_error = None
        text_correct = coverage = None
        if task is not None:
            judge_tags = {"run": run_id, "qid": question.id}
            try:
                judge = get_judge()
                if task == "text":
                    result = await judge.judge_text(question, output.answer, tags=judge_tags)
                    text_correct = result.correct
                else:
                    result = await judge.judge_key_points(question, output.answer, tags=judge_tags)
                    coverage = key_point_coverage(question, result)
                verdict = result.model_dump(mode="json")
            except Exception as exc:  # judge failure: metric stays null, question stays ok
                judge_error = f"{type(exc).__name__}: {exc}"
                text_correct = coverage = None

        metrics = score_question(
            question, output, text_correct=text_correct, key_point_coverage=coverage
        )
    return _QuestionRun(
        question=question,
        output=output,
        status="ok" if error is None else "error",
        error=error,
        started_at=started_at,
        ended_at=ended_at,
        latency_ms=latency_ms,
        judge_task=task,
        judge_verdict=verdict,
        judge_error=judge_error,
        metrics=metrics,
    )


def _is_judge(row: dict[str, Any]) -> bool:
    return row.get("tags", {}).get("role") == "judge"


def _retrieved_summary(output: PipelineOutput | None) -> list[dict[str, Any]]:
    if output is None:
        return []
    return [
        {
            "rank": rank,
            "chunk_id": item.chunk_id,
            "doc": item.doc,
            "page_start": item.page_start,
            "page_end": item.page_end,
            "score": item.score,
            "text_chars": item.text_chars,
        }
        for rank, item in enumerate(output.retrieved, start=1)
    ]


def _result_row(run: _QuestionRun, calls: list[dict[str, Any]]) -> dict[str, Any]:
    """One `results.jsonl` row.

    Cost/token/call fields cover the pipeline's LLM calls only; `judge_*` fields cover
    the judge's. `latency_ms` is the pipeline's wall time for this question;
    `latency_eligible` is false when any pipeline LLM call was a cache hit.
    """
    q = run.question
    pipeline_calls = [c for c in calls if not _is_judge(c)]
    judge_calls = [c for c in calls if _is_judge(c)]
    return {
        "id": q.id,
        "type": q.type,
        "split": q.split,
        "pilot": q.pilot,
        "verified": q.verified,
        "expected_behavior": q.expected_behavior,
        "status": run.status,
        "error": run.error,
        "answer": None if run.output is None else run.output.answer.model_dump(mode="json"),
        "retrieved": _retrieved_summary(run.output),
        "metrics": run.metrics,
        "judge_task": run.judge_task,
        "judge_verdict": run.judge_verdict,
        "judge_error": run.judge_error,
        "latency_ms": round(run.latency_ms, 1),
        "latency_eligible": run.status == "ok" and not any(c.get("cached") for c in pipeline_calls),
        "cost_usd": round(float(sum(c.get("cost_usd") or 0 for c in pipeline_calls)), 8),
        "input_tokens": sum(_prompt_tokens(c) for c in pipeline_calls),
        "output_tokens": sum(c.get("output_tokens") or 0 for c in pipeline_calls),
        "llm_calls": len(pipeline_calls),
        "cached_llm_calls": sum(bool(c.get("cached")) for c in pipeline_calls),
        "judge_cost_usd": round(float(sum(c.get("cost_usd") or 0 for c in judge_calls)), 8),
        "judge_llm_calls": len(judge_calls),
    }


def _prompt_tokens(row: dict[str, Any]) -> int:
    return sum(
        row.get(k) or 0
        for k in ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")
    )


def _trace_record(run: _QuestionRun, calls: list[dict[str, Any]], run_id: str) -> dict[str, Any]:
    """One `traces.jsonl` record: everything about one question, self-contained.

    Maps to one Langfuse trace: `retrieval` -> a retrieval span; each `llm_calls` entry
    (a verbatim `kaprag.llm` trace row plus `role`: "pipeline" | "judge") -> one
    generation; `answer` -> the trace output.
    """
    return {
        "run_id": run_id,
        "qid": run.question.id,
        "question": run.question.question,
        "type": run.question.type,
        "status": run.status,
        "error": run.error,
        "started_at": run.started_at,
        "ended_at": run.ended_at,
        "latency_ms": round(run.latency_ms, 1),
        "retrieval": _retrieved_summary(run.output),
        "answer": None if run.output is None else run.output.answer.model_dump(mode="json"),
        "llm_calls": [{"role": "judge" if _is_judge(c) else "pipeline", **c} for c in calls],
        "judge": {"task": run.judge_task, "verdict": run.judge_verdict, "error": run.judge_error},
    }


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _display_path(path: Path) -> str:
    path = Path(path).resolve()
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()
