"""Langfuse exporter: JSONL run folder -> traces, observations, scores (no SDK, no network)."""

import re
import shutil
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from kaprag.observability import export_cli
from kaprag.observability.langfuse_mapping import (
    load_run,
    observation_id_for,
    trace_id_for,
)

FIXTURE = Path(__file__).parent / "fixtures" / "langfuse" / "2026-01-02-router-dev"
SRC = Path(__file__).resolve().parents[1] / "src" / "kaprag"
CRED_VARS = ("LANGFUSE_HOST", "LANGFUSE_BASE_URL", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY")


def ts(text: str) -> datetime:
    return datetime.fromisoformat(text)


class FakeSink:
    """Records what the exporter sends; `existing` simulates traces already in Langfuse."""

    def __init__(self, existing=None, fail_on=(), reject_scores=False):
        self.existing = set(existing or ())
        self.fail_on = set(fail_on)
        self.reject_scores = reject_scores
        self.traces = []
        self.scores = []
        self.flushed = 0
        self.closed = 0

    def trace_exists(self, trace_id):
        return trace_id in self.existing

    def send_trace(self, trace):
        if trace.qid in self.fail_on:
            raise RuntimeError("export failed: HTTP 500")
        self.traces.append(trace)
        self.existing.add(trace.id)

    def send_scores(self, scores):
        if self.reject_scores:
            raise RuntimeError("2 scores rejected")
        self.scores.extend(scores)

    def flush(self):
        self.flushed += 1

    def close(self):
        self.closed += 1


@pytest.fixture
def run():
    return load_run(FIXTURE)


def by_qid(run, qid):
    return next(t for t in run.traces if t.qid == qid)


def named(trace, name, kind=None):
    [obs] = [o for o in trace.observations if o.name == name and kind in (None, o.type)]
    return obs


def children(trace, parent):
    return [o for o in trace.observations if o.parent_id == parent.id]


# --- mapping ---------------------------------------------------------------


def test_one_trace_per_question_with_run_config_metadata(run):
    assert run.run_key == "2026-01-02-router-dev"
    assert [t.qid for t in run.traces] == ["q001", "q002", "q003"]
    assert run.errors == []
    t = by_qid(run, "q001")
    assert t.id == trace_id_for("2026-01-02-router-dev", "q001")
    assert t.name == "router"
    assert t.session_id == "2026-01-02-router-dev"
    assert set(t.tags) >= {
        "run:2026-01-02-router-dev", "config:router", "split:dev", "type:olgusal", "status:ok",
        "route:simple",
    }  # fmt: skip
    md = t.metadata
    assert md["config_name"] == "router"
    assert md["split"] == "dev"
    assert md["git_commit"] == "9f1c2e7a0b3d4c5e6f708192a3b4c5d6e7f80912"
    assert md["questions_sha256"].endswith("abc")
    assert md["pipeline_factory"] == "router"
    assert md["pipeline_params"]["decompose"] == {"model": "claude-sonnet-5-5", "max_lookups": 8}
    assert md["judge_model"] == "claude-sonnet-5-5"
    assert md["qid"] == "q001"
    assert md["question_type"] == "olgusal"
    assert md["verified"] is True


def test_root_observation_carries_question_answer_and_jsonl_timestamps(run):
    t = by_qid(run, "q001")
    root = t.root
    assert root.parent_id is None
    assert root is t.observations[0]
    assert root.input == {"question": "Akbank 2023 ikinci asama gecikme gun sayisi nedir?"}
    assert root.output["answer"].startswith("Gecikme 30")
    assert root.start == ts("2026-01-02T10:00:00+00:00")
    # The judge runs after the pipeline; the root covers it.
    assert root.end == ts("2026-01-02T10:00:05+00:00")
    assert root.level == "DEFAULT"
    assert root.metadata["pipeline_latency_ms"] == 3000.0


def test_retrieval_span_lists_chunks_with_scores(run):
    t = by_qid(run, "q001")
    retrieval = named(t, "retrieval")
    assert retrieval.type == "retriever"
    assert retrieval.parent_id == t.root.id
    assert retrieval.output == [
        {"rank": 1, "chunk_id": "akbank-2023:fixed:655", "doc": "akbank-2023",
         "page_start": 212, "page_end": 212, "score": 0.998, "text_chars": 2000},
        {"rank": 2, "chunk_id": "akbank-2024:fixed:664", "doc": "akbank-2024",
         "page_start": 223, "page_end": 224, "score": 0.996, "text_chars": 2000},
    ]  # fmt: skip
    assert retrieval.metadata["context_chunk_ids"] == ["akbank-2023:fixed:655"]
    assert retrieval.metadata["retrieval_trace"][0]["fts_rank"] == 2
    # Bounds inferred: after the classify call, up to the start of the generate call.
    assert retrieval.start == ts("2026-01-02T10:00:01+00:00")
    assert retrieval.end == ts("2026-01-02T10:00:03+00:00") - timedelta(milliseconds=12.5)
    assert retrieval.metadata["timing"] == "inferred"


def test_one_generation_per_llm_call_with_usage_cost_cache_and_timestamps(run):
    t = by_qid(run, "q001")
    gens = [o for o in t.observations if o.type == "generation"]
    assert [g.name for g in gens] == ["classify", "generate", "judge:text"]
    gen = named(t, "generate")
    assert gen.parent_id == t.root.id
    assert gen.model == "claude-haiku-5-5"
    assert gen.end == ts("2026-01-02T10:00:03+00:00")
    assert gen.start == gen.end - timedelta(milliseconds=12.5)
    assert gen.usage_details == {
        "input": 2, "output": 300, "cache_read_input_tokens": 900,
        "cache_creation_input_tokens": 7000,
    }  # fmt: skip
    assert gen.cost_details == {"total": 0.0011}
    assert gen.input == {"prompt_hash": gen.metadata["prompt_hash"]}
    assert gen.metadata["cached"] is True
    assert gen.metadata["source_latency_ms"] == 900.0
    assert gen.metadata["role"] == "pipeline"
    assert gen.metadata["step"] == "generate"
    assert gen.level == "DEFAULT"


def test_judge_calls_sit_under_an_evaluator_span_apart_from_pipeline_calls(run):
    t = by_qid(run, "q001")
    judge = named(t, "judge")
    assert judge.type == "evaluator"
    assert judge.parent_id == t.root.id
    assert judge.input == {"task": "text"}
    assert judge.output == {"reason": "matches the reference", "correct": True}
    [jgen] = children(t, judge)
    assert jgen.type == "generation"
    assert jgen.name == "judge:text"
    assert jgen.metadata["role"] == "judge"
    assert jgen.model == "claude-sonnet-5-5"
    assert (judge.start, judge.end) == (jgen.start, jgen.end)
    pipeline_gens = [o for o in t.observations if o.type == "generation" and o is not jgen]
    assert all(g.metadata["role"] == "pipeline" for g in pipeline_gens)


def test_router_debug_becomes_nested_spans(run):
    t = by_qid(run, "q002")
    router = named(t, "router")
    assert router.type == "agent"
    assert router.parent_id == t.root.id
    assert router.output == {"route": "multi"}
    steps = children(t, router)
    assert [s.name for s in steps] == [
        "classify", "decompose", "lookup:a", "lookup:b", "compute", "synthesize",
    ]  # fmt: skip
    decompose = named(t, "decompose", "span")
    assert decompose.output["lookups"][1]["bank"] == "garanti"
    [dgen] = children(t, decompose)
    assert dgen.name == "decompose" and dgen.type == "generation"

    lookup_a = named(t, "lookup:a")
    assert lookup_a.input["sub_question"] == "Akbank 2023 sube sayisi?"
    assert lookup_a.input["filter"] == {"banks": ["akbank"], "years": [2023]}
    assert lookup_a.output["value"] == 700
    assert lookup_a.metadata["retrieved"][0]["chunk_id"] == "akbank-2023:fixed:1"
    [agen] = children(t, lookup_a)
    assert agen.name == "sub_answer:a"
    assert (lookup_a.start, lookup_a.end) == (agen.start, agen.end)

    compute = named(t, "compute")
    assert compute.type == "tool"
    assert compute.output == {"kind": "min", "value": 700, "operand": "a", "error": None}
    assert children(t, compute) == []
    lookup_b = named(t, "lookup:b")
    assert compute.start == compute.end == lookup_b.end  # no timing recorded: instant
    assert router.start == named(t, "classify", "span").start
    assert router.end == named(t, "synthesize", "span").end

    gens = [o for o in t.observations if o.type == "generation"]
    assert len(gens) == 6
    assert len(t.observations) == 16


def test_failed_question_and_failed_call_are_marked_error(run):
    t = by_qid(run, "q003")
    assert t.root.level == "ERROR"
    assert t.root.status_message == "RuntimeError: classify failed"
    assert t.root.output is None
    assert "status:error" in t.tags
    [gen] = [o for o in t.observations if o.type == "generation"]
    assert gen.level == "ERROR"
    assert gen.status_message == "TimeoutError: backend timed out"
    # No pipeline output: no retrieval span, the call hangs off the root.
    assert [o.name for o in t.observations] == ["q003", "classify"]


def test_results_metrics_become_scores(run):
    t1 = by_qid(run, "q001")
    s1 = {s.name: s for s in run.scores if s.trace_id == t1.id}
    assert set(s1) == {
        "answer_accuracy", "page_recall@5", "page_recall@10", "mrr@10", "text_correctness",
        "citation_precision", "citation_recall", "behavior_accuracy", "false_abstain_rate",
    }  # fmt: skip
    text = s1["text_correctness"]
    assert (text.value, text.data_type) == (1.0, "BOOLEAN")
    assert text.comment == "matches the reference"
    assert (s1["mrr@10"].value, s1["mrr@10"].data_type) == (1.0, "NUMERIC")
    assert s1["false_abstain_rate"].value == 0.0
    assert text.id == f"{t1.id}-text_correctness"
    assert text.timestamp == ts("2026-01-02T10:00:03+00:00")
    assert text.metadata == {"field": "text_correct", "content_metric": "text"}

    t2 = by_qid(run, "q002")
    s2 = {s.name: s for s in run.scores if s.trace_id == t2.id}
    assert s2["numeric_accuracy"].value == 0.0
    assert s2["numeric_accuracy"].comment == "outside tolerance: 700 vs 705 (abs 0.5)"
    assert "text_correctness" not in s2
    assert len(run.scores) == 9 + 10 + 5


# --- ids ---------------------------------------------------------------------


def all_ids(run):
    return (
        [t.id for t in run.traces],
        [o.id for t in run.traces for o in t.observations],
        [s.id for s in run.scores],
    )


def test_ids_are_deterministic_unique_and_well_formed(run):
    traces, observations, scores = all_ids(run)
    assert all_ids(load_run(FIXTURE)) == (traces, observations, scores)
    assert all(re.fullmatch(r"[0-9a-f]{32}", i) for i in traces)
    assert all(re.fullmatch(r"[0-9a-f]{16}", i) for i in observations)
    assert len(set(traces)) == len(traces)
    assert len(set(observations)) == len(observations)
    assert len(set(scores)) == len(scores)
    assert observations[0] == observation_id_for("2026-01-02-router-dev", "q001", "root")
    # Same scheme as Langfuse.create_trace_id(seed=...): sha256(seed)[:16 bytes].hex()
    assert trace_id_for("r", "q") == "136dfb09cf2288fcbe07223a2b46599f"


def test_different_run_folder_gives_different_ids(run, tmp_dir):
    other = tmp_dir / "2026-01-03-router-dev"
    shutil.copytree(FIXTURE, other)
    traces, observations, scores = all_ids(load_run(other))
    a_traces, a_observations, a_scores = all_ids(run)
    assert not set(traces) & set(a_traces)
    assert not set(observations) & set(a_observations)
    assert not set(scores) & set(a_scores)


# --- CLI -----------------------------------------------------------------------


def no_sink():
    raise AssertionError("dry run must not create a sink")


def test_dry_run_prints_counts_without_a_sink(capsys):
    assert export_cli.main(["--run", str(FIXTURE), "--dry-run"], sink_factory=no_sink) == 0
    out = capsys.readouterr().out
    assert "dry run" in out
    assert "traces=3 observations=26 generations=10 scores=24 errors=0" in out


def test_limit(capsys):
    assert export_cli.main(["--run", str(FIXTURE), "--dry-run", "--limit", "1"]) == 0
    assert "traces=1 observations=8 generations=3 scores=9" in capsys.readouterr().out


def test_export_sends_everything_and_second_export_adds_no_traces(capsys):
    sink = FakeSink()
    assert export_cli.main(["--run", str(FIXTURE)], sink_factory=lambda: sink) == 0
    assert [t.qid for t in sink.traces] == ["q001", "q002", "q003"]
    assert len(sink.scores) == 24
    assert sink.flushed == 1 and sink.closed == 1
    out = capsys.readouterr().out
    assert "sent=3 skipped_existing=0 failed=0 scores=24 errors=0" in out

    again = FakeSink(existing=sink.existing)
    assert export_cli.main(["--run", str(FIXTURE)], sink_factory=lambda: again) == 0
    assert again.traces == []
    # Scores are re-sent with the same ids and timestamps: Langfuse overwrites them.
    assert [s.id for s in again.scores] == [s.id for s in sink.scores]
    assert [s.timestamp for s in again.scores] == [s.timestamp for s in sink.scores]
    assert "sent=0 skipped_existing=3 failed=0" in capsys.readouterr().out


def test_send_failure_continues_and_exits_nonzero(capsys):
    sink = FakeSink(fail_on={"q002"})
    assert export_cli.main(["--run", str(FIXTURE)], sink_factory=lambda: sink) == 1
    assert [t.qid for t in sink.traces] == ["q001", "q003"]
    assert {s.trace_id for s in sink.scores} == {sink.traces[0].id, sink.traces[1].id}
    captured = capsys.readouterr()
    assert "q002" in captured.err and "HTTP 500" in captured.err
    assert "sent=2 skipped_existing=0 failed=1" in captured.out
    assert sink.flushed == 1 and sink.closed == 1


def test_rejected_scores_are_reported_and_exit_nonzero(capsys):
    sink = FakeSink(reject_scores=True)
    assert export_cli.main(["--run", str(FIXTURE)], sink_factory=lambda: sink) == 1
    assert len(sink.traces) == 3
    captured = capsys.readouterr()
    assert "q001 scores: RuntimeError: 2 scores rejected" in captured.err
    assert "sent=3 skipped_existing=0 failed=0 scores=0 errors=3" in captured.out


def test_malformed_trace_line_is_reported_and_the_rest_exported(tmp_dir, capsys):
    run_dir = tmp_dir / "2026-01-02-router-dev"
    shutil.copytree(FIXTURE, run_dir)
    path = run_dir / "traces.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    lines.insert(1, '{"qid": "q999", "broken": ')
    lines.insert(3, '{"run_id": "x", "qid": "q998"}')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    sink = FakeSink()
    assert export_cli.main(["--run", str(run_dir)], sink_factory=lambda: sink) == 1
    assert [t.qid for t in sink.traces] == ["q001", "q002", "q003"]
    captured = capsys.readouterr()
    assert "traces.jsonl:2:" in captured.err
    assert "traces.jsonl:4:" in captured.err
    assert "errors=2" in captured.out

    assert export_cli.main(["--run", str(run_dir), "--dry-run"]) == 1
    assert "traces=3 observations=26 generations=10 scores=24 errors=2" in capsys.readouterr().out


def test_missing_credentials_is_a_clear_error_without_network(monkeypatch, capsys):
    for var in CRED_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-should-not-be-printed")
    assert export_cli.main(["--run", str(FIXTURE)]) == 2
    err = capsys.readouterr().err
    assert "LANGFUSE_HOST" in err and "LANGFUSE_SECRET_KEY" in err
    assert "LANGFUSE_PUBLIC_KEY" not in err.split("missing")[-1]
    assert "pk-lf-should-not-be-printed" not in err


def test_missing_run_folder_is_a_usage_error(tmp_dir, capsys):
    assert export_cli.main(["--run", str(tmp_dir / "nope"), "--dry-run"]) == 2
    assert "config.yaml" in capsys.readouterr().err


def test_only_the_sink_module_imports_the_langfuse_sdk():
    pattern = re.compile(r"^\s*(import langfuse|from langfuse)", re.MULTILINE)
    importers = [
        p.relative_to(SRC).as_posix()
        for p in SRC.rglob("*.py")
        if pattern.search(p.read_text(encoding="utf-8"))
    ]
    assert importers == ["observability/langfuse_sink.py"]
