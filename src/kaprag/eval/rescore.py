"""Re-score a finished run's STORED outputs against the current gold file.

The gold file changes (audit fixes, rejections, evidence pages) but measured system
outputs are never regenerated: the test split was run exactly once and the index has
changed since. Re-scoring reads each question's stored structured answer and
retrieved list from the original `results.jsonl` (question text from `traces.jsonl`)
and scores it against the CURRENT gold row with the runner's own scoring
(`runner.score_output`, `runner.result_row`, `summary.summarize`). No pipeline is
built, nothing is retrieved, no model is loaded.

Judge calls go through `kaprag.llm` with the normal disk cache: an unchanged
(question, expected answer, system answer) prompt is a cache hit, so only questions
whose reference answer or key points changed cost a fresh call. Operational fields
(cost, tokens, latency and its cache exclusion) are carried over from the original
per-question rows, not re-measured.

Questions of the original run no longer in the gold file are dropped; questions whose
text differs from the stored text are excluded (the stored answer belongs to the old
wording); gold questions of the split the run did not contain are only reported.

Output: a NEW folder `<runs_dir>/<original-name>-rescored[-N]/` with `README.md`,
`config.yaml` (original config + a `rescore` block), `results.jsonl` and
`summary.json`. The `run` block of both names the CURRENT gold file, its sha256 and
the re-scored question count; the original values are in the `rescore` block. The
folder is created only after scoring succeeded. The original folder is only read.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from kaprag.eval.config import EvalConfig
from kaprag.eval.pipeline import PipelineOutput
from kaprag.eval.runner import (
    DEFAULT_RUNS_DIR,
    OPERATIONAL_FIELDS,
    LLMFactory,
    _default_llm_factory,
    _display_path,
    _LazyLLM,
    _write_jsonl,
    git_commit,
    is_judge_call,
    judge_summary,
    make_judge_getter,
    result_row,
    score_output,
    select_questions,
    sha256_file,
)
from kaprag.eval.schema import Question, load_questions
from kaprag.eval.summary import summarize

RUN_FILES = ("config.yaml", "summary.json", "results.jsonl", "traces.jsonl")


class RescoreError(ValueError):
    """The run folder cannot be re-scored (missing files, already a re-scoring, ...)."""


@dataclass
class RescoreResult:
    run_dir: Path
    summary: dict[str, Any]


@dataclass
class _Original:
    run_dir: Path
    config: dict[str, Any]
    summary: dict[str, Any]
    rows: list[dict[str, Any]]
    question_text: dict[str, str]


def _read_original(run_dir: Path) -> _Original:
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise RescoreError(f"run folder not found: {run_dir}")
    config_path = run_dir / "config.yaml"
    config = (
        yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    )
    if "rescore" in config:
        raise RescoreError(f"{run_dir.name} is already a re-scoring; re-score the original run")
    missing = [name for name in RUN_FILES if not (run_dir / name).is_file()]
    if missing:
        raise RescoreError(f"{run_dir} lacks {', '.join(missing)}; cannot re-score it")
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    rows = _read_jsonl(run_dir / "results.jsonl")
    traces = _read_jsonl(run_dir / "traces.jsonl")
    text = {t["qid"]: t["question"] for t in traces}
    if lost := [r["id"] for r in rows if r["id"] not in text]:
        raise RescoreError(f"traces.jsonl has no question text for {lost}")
    return _Original(run_dir, config, summary, rows, text)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _stored_output(row: dict[str, Any]) -> PipelineOutput | None:
    """The pipeline output as the runner scored it (None for a failed question)."""
    if row["status"] == "error":
        return None
    return PipelineOutput.model_validate({"answer": row["answer"], "retrieved": row["retrieved"]})


def _make_dir(runs_dir: Path, base: str) -> Path:
    runs_dir.mkdir(parents=True, exist_ok=True)
    n = 1
    while True:
        path = runs_dir / (base if n == 1 else f"{base}-{n}")
        try:
            path.mkdir()
            return path
        except FileExistsError:
            n += 1


async def rescore_run_async(
    run_dir: Path,
    *,
    questions_path: Path,
    runs_dir: Path = DEFAULT_RUNS_DIR,
    llm_factory: LLMFactory = _default_llm_factory,
    today: date | None = None,
) -> RescoreResult:
    """Re-score one run folder against `questions_path`; writes a new folder."""
    original = _read_original(Path(run_dir))
    run_meta = original.config.get("run") or {}
    config = EvalConfig.model_validate({k: v for k, v in original.config.items() if k != "run"})
    questions = load_questions(questions_path)
    gold = {q.id: q for q in questions}

    kept: list[tuple[Question, dict[str, Any]]] = []
    dropped, excluded = [], []
    for row in original.rows:
        q = gold.get(row["id"])
        if q is None:
            dropped.append(row["id"])
        elif q.question != original.question_text[row["id"]]:
            excluded.append(row["id"])
        else:
            kept.append((q, row))
    in_run = {r["id"] for r in original.rows}
    split = run_meta.get("split")
    missing = (
        [q.id for q in select_questions(questions, split) if q.id not in in_run] if split else []
    )

    base_name = f"{original.run_dir.name}-rescored"
    with tempfile.TemporaryDirectory(prefix="kaprag-rescore-") as tmp:
        llm = _LazyLLM(llm_factory, Path(tmp) / "llm-trace.jsonl")
        get_judge = make_judge_getter(config, llm)
        semaphore = asyncio.Semaphore(config.concurrency)

        async def score(q: Question, row: dict[str, Any]):
            async with semaphore:
                return await score_output(
                    q, _stored_output(row), get_judge, tags={"run": base_name, "qid": q.id}
                )

        scored = await asyncio.gather(*(score(q, row) for q, row in kept))
        trace_rows = llm.rows()

    calls_by_qid: dict[str, list[dict[str, Any]]] = {}
    for call in trace_rows:
        calls_by_qid.setdefault(call.get("tags", {}).get("qid"), []).append(call)
    results = []
    for (q, row), s in zip(kept, scored, strict=True):
        calls = sorted(calls_by_qid.get(q.id, []), key=lambda c: c.get("ts", ""))
        results.append(
            result_row(
                q,
                status=row["status"],
                error=row["error"],
                answer=row["answer"],
                retrieved=row["retrieved"],
                scored=s,
                operational={k: row[k] for k in OPERATIONAL_FIELDS},
                judge_calls=[c for c in calls if is_judge_call(c)],
            )
        )

    judge_calls = [c for c in trace_rows if is_judge_call(c)]
    fresh = sum(not c.get("cached") for c in judge_calls)
    current_sha = sha256_file(questions_path)
    rescore_meta = {
        "note": "re-scoring of stored outputs against the current gold file; no pipeline was run",
        "original_run_id": original.run_dir.name,
        "original_questions_path": run_meta.get("questions_path"),
        "original_questions_sha256": run_meta.get("questions_sha256"),
        "original_n_questions": run_meta.get("n_questions"),
        "questions_path": _display_path(questions_path),
        "questions_sha256": current_sha,
        "date": (today or date.today()).isoformat(),
        "git_commit": git_commit(),
        "n_original": len(original.rows),
        "n_scored": len(results),
        "dropped_not_in_gold": dropped,
        "excluded_question_changed": excluded,
        "missing_from_run": missing,
        "judge_calls_fresh": fresh,
        "judge_calls_cached": len(judge_calls) - fresh,
    }
    # The run block names the gold version and question count actually scored against
    # (consumers such as the Langfuse export read it); the original values stay in the
    # rescore block.
    current = {
        "questions_path": rescore_meta["questions_path"],
        "questions_sha256": current_sha,
        "n_questions": len(results),
    }
    summary = {
        "run": {**original.summary.get("run", {}), **current},
        "rescore": rescore_meta,
        **summarize(results),
        "judge": judge_summary(config, results)
        | {"llm_calls_fresh": fresh, "llm_calls_cached": len(judge_calls) - fresh},
        "unattributed_llm_calls": original.summary.get("unattributed_llm_calls"),
    }

    config_out = {**original.config, "run": {**run_meta, **current}, "rescore": rescore_meta}
    # Created only now, after scoring succeeded; removed again if writing fails.
    new_dir = _make_dir(Path(runs_dir), base_name)
    try:
        _write_folder(new_dir, original.run_dir.name, config_out, results, summary)
    except BaseException:
        shutil.rmtree(new_dir, ignore_errors=True)
        raise
    return RescoreResult(run_dir=new_dir, summary=summary)


def _write_folder(
    new_dir: Path,
    original_name: str,
    config_out: dict[str, Any],
    results: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    block = summary["rescore"]
    (new_dir / "README.md").write_text(
        f"Re-scoring of the stored outputs of runs/{original_name} (no pipeline run) "
        f"against {block['questions_path']} sha256 {block['questions_sha256'][:12]}; "
        f"traces: runs/{original_name}/traces.jsonl\n",
        encoding="utf-8",
    )
    (new_dir / "config.yaml").write_text(
        yaml.safe_dump(
            config_out,
            allow_unicode=True,
            sort_keys=False,
            width=10**6,
        ),
        encoding="utf-8",
    )
    _write_jsonl(new_dir / "results.jsonl", results)
    (new_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def rescore_run(run_dir: Path, **kwargs: Any) -> RescoreResult:
    """Blocking wrapper around `rescore_run_async`."""
    return asyncio.run(rescore_run_async(run_dir, **kwargs))


def format_report(result: RescoreResult, ids_limit: int = 20) -> Sequence[str]:
    """Console lines for a finished re-scoring."""
    block = result.summary["rescore"]

    def ids(values: list[str]) -> str:
        shown = ", ".join(values[:ids_limit])
        return f"{len(values)}" + (
            f" ({shown}{', ...' if len(values) > ids_limit else ''})" if values else ""
        )

    lines = [
        f"note: {block['note']}",
        f"original run: {block['original_run_id']} -> {result.run_dir}",
        f"gold sha256: {block['original_questions_sha256']} -> {block['questions_sha256']}",
        f"questions: {block['n_original']} in the run, {block['n_scored']} scored",
        f"dropped (no longer in gold): {ids(block['dropped_not_in_gold'])}",
        f"excluded (question text changed): {ids(block['excluded_question_changed'])}",
        f"in the gold split but not in the run (not scored): {ids(block['missing_from_run'])}",
        f"judge calls: fresh {block['judge_calls_fresh']}, cached {block['judge_calls_cached']}",
    ]
    if (result.summary.get("run") or {}).get("split") == "test":
        lines.insert(0, "test split: this is a re-scoring of stored outputs, not a new test run")
    return lines
