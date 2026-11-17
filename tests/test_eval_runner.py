"""Eval runner: run folder, per-question isolation, LLM cost/latency attribution."""

import hashlib
import json
import re
from datetime import date
from pathlib import Path

import pytest
import yaml

from kaprag.eval.config import EvalConfig, load_config
from kaprag.eval.fake_pipeline import FakeOraclePipeline
from kaprag.eval.judge import KEY_POINTS_SYSTEM, TEXT_SYSTEM
from kaprag.eval.pipeline import register_pipeline, resolve_factory
from kaprag.eval.runner import make_run_dir, run_eval, select_questions
from kaprag.eval.schema import load_questions
from kaprag.llm import LLMClient, Usage, pricing
from kaprag.llm.types import BackendResponse

FIXTURE = Path(__file__).parent / "fixtures" / "eval" / "questions.jsonl"
QUESTIONS = load_questions(FIXTURE)
RUN_FILES = {"README.md", "config.yaml", "results.jsonl", "summary.json", "traces.jsonl"}
DAY = date(2026, 11, 17)


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def config(factory="fake_oracle", params=None, judge="fake"):
    return EvalConfig(
        description="test run",
        topic="unit",
        pipeline={"factory": factory, "params": params or {}},
        judge={"kind": judge},
        concurrency=3,
    )


def run(cfg, tmp_dir, split="dev", **kwargs):
    return run_eval(
        cfg,
        config_name="unit",
        questions=QUESTIONS,
        questions_path=FIXTURE,
        split=split,
        runs_dir=tmp_dir / "runs",
        today=DAY,
        **kwargs,
    )


# --- selection and folders -------------------------------------------------


def test_select_questions():
    assert [q.id for q in select_questions(QUESTIONS, "test")] == ["f010", "f011"]
    pilot = [q.id for q in select_questions(QUESTIONS, "pilot")]
    assert pilot == ["f001", "f002", "f005", "f006", "f007", "f009", "f011"]
    assert len(select_questions(QUESTIONS, "dev")) == 11
    assert [q.id for q in select_questions(QUESTIONS, "dev", limit=2)] == ["f001", "f002"]
    with pytest.raises(ValueError):
        select_questions(QUESTIONS, "train")


def test_make_run_dir_never_reuses(tmp_dir):
    first = make_run_dir(tmp_dir, "baseline", "dev", DAY)
    second = make_run_dir(tmp_dir, "baseline", "dev", DAY)
    third = make_run_dir(tmp_dir, "baseline", "dev", DAY)
    assert [p.name for p in (first, second, third)] == [
        "2026-11-17-baseline-dev",
        "2026-11-17-baseline-dev-2",
        "2026-11-17-baseline-dev-3",
    ]


def test_resolve_factory():
    assert resolve_factory("fake_oracle") is resolve_factory(
        "kaprag.eval.fake_pipeline:build_fake_oracle"
    )
    with pytest.raises(ValueError, match="unknown pipeline"):
        resolve_factory("nope")
    with pytest.raises(ValueError, match="not found"):
        resolve_factory("kaprag.eval.fake_pipeline:nope")


def test_shipped_fake_config_is_valid_and_fake():
    cfg = load_config("fake-oracle")
    assert cfg.pipeline.factory == "fake_oracle"
    assert cfg.judge.kind == "fake"
    assert "FAKE" in cfg.description


# --- fake pipeline end to end ----------------------------------------------


def test_run_writes_exactly_the_five_files(tmp_dir):
    result = run(config(params={"error_rate": 0.3, "fail_rate": 0.2, "seed": 1}), tmp_dir)
    d = result.run_dir
    assert d.name == "2026-11-17-unit-dev"
    assert {p.name for p in d.iterdir()} == RUN_FILES
    assert d.joinpath("README.md").read_text(encoding="utf-8") == "test run\n"

    cfg = yaml.safe_load(d.joinpath("config.yaml").read_text(encoding="utf-8"))
    assert cfg["pipeline"]["factory"] == "fake_oracle"
    meta = cfg["run"]
    assert meta["split"] == "dev"
    assert meta["questions_path"] == "tests/fixtures/eval/questions.jsonl"
    assert meta["questions_sha256"] == hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
    assert re.fullmatch(r"[0-9a-f]{40}(-dirty)?", meta["git_commit"])
    assert meta["started_at"]

    results = read_jsonl(d / "results.jsonl")
    traces = read_jsonl(d / "traces.jsonl")
    dev = [q for q in QUESTIONS if q.split == "dev"]
    assert [r["id"] for r in results] == [q.id for q in dev]
    assert [t["qid"] for t in traces] == [q.id for q in dev]
    assert [r["verified"] for r in results] == [q.verified for q in dev]
    assert {r["status"] for r in results} == {"ok", "error"}  # seed 1 fails some

    summary = json.loads(d.joinpath("summary.json").read_text(encoding="utf-8"))
    assert summary["overall"]["n"] == len(dev)
    assert summary["overall"]["n_error"] == sum(r["status"] == "error" for r in results)
    assert summary["verified"]["n"] == sum(q.verified for q in dev)
    assert set(summary["by_type"]) >= {"olgusal", "sentez", "belirsiz"}
    assert summary["run"]["run_id"] == d.name


def test_error_rate_zero_scores_perfectly(tmp_dir):
    result = run(config(params={"error_rate": 0, "fail_rate": 0}), tmp_dir)
    metrics = result.summary["overall"]["metrics"]
    assert metrics["answer_accuracy"] == {"value": 1.0, "n": 11}
    assert metrics["page_recall@10"]["value"] == 1.0
    assert metrics["all_evidence_recall@10"]["value"] == 1.0
    assert metrics["citation_precision"]["value"] == 1.0
    assert metrics["false_abstain_rate"]["value"] == 0.0
    assert metrics["correct_abstain_rate"]["value"] == 1.0


def test_every_question_failing_still_completes(tmp_dir):
    result = run(config(params={"fail_rate": 1.0}), tmp_dir, limit=4)
    overall = result.summary["overall"]
    assert (overall["n"], overall["n_error"]) == (4, 4)
    assert overall["metrics"]["answer_accuracy"] == {"value": 0.0, "n": 4}
    rows = read_jsonl(result.run_dir / "results.jsonl")
    assert rows[0]["error"].startswith("FakePipelineFailure: ")
    assert rows[0]["answer"] is None and rows[0]["retrieved"] == []


def test_runs_never_overwrite(tmp_dir):
    first = run(config(), tmp_dir, limit=1).run_dir
    second = run(config(), tmp_dir, limit=1).run_dir
    assert second.name == first.name + "-2"
    assert {p.name for p in first.iterdir()} == RUN_FILES


def test_empty_selection_creates_no_folder(tmp_dir):
    only_dev = [q for q in QUESTIONS if q.split == "dev"]
    with pytest.raises(ValueError, match="no questions"):
        run_eval(
            config(),
            config_name="unit",
            questions=only_dev,
            questions_path=FIXTURE,
            split="test",
            runs_dir=tmp_dir / "runs",
        )
    assert not (tmp_dir / "runs").exists()


# --- LLM attribution: cost, cache hits, judge separation --------------------

PIPELINE_USAGE = Usage(input_tokens=1000, output_tokens=100)
JUDGE_USAGE = Usage(input_tokens=200, output_tokens=20)


class ScriptedBackend:
    """Answers pipeline prompts with text and judge prompts with a verdict JSON."""

    name = "fake"

    def __init__(self, judge_reply=None):
        self.judge_reply = judge_reply
        self.calls = 0

    async def generate(self, system, user, model):
        self.calls += 1
        if system == TEXT_SYSTEM:
            text = self.judge_reply or '{"reason": "ok", "correct": true}'
            return BackendResponse(text=text, usage=JUDGE_USAGE)
        if system == KEY_POINTS_SYSTEM:
            n = int(re.search(r"KEY POINTS \((\d+)\)", user).group(1))
            text = json.dumps({"reason": "ok", "covered": [True] * n})
            return BackendResponse(text=text, usage=JUDGE_USAGE)
        return BackendResponse(text="draft", usage=PIPELINE_USAGE)


class LLMEchoPipeline:
    """Makes one tagged LLM call per question, then answers like a perfect oracle."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.oracle = FakeOraclePipeline(ctx.gold, error_rate=0, fail_rate=0, seed=0)
        self.closed = False

    async def run(self, question, *, tags):
        if self.ctx.params.get("raise_on") == tags["qid"]:
            raise KeyError("not an LLM error")
        call_tags = {} if self.ctx.params.get("untagged") else {**tags, "step": "generate"}
        await self.ctx.llm().complete("echo", question, "claude-haiku-5-5", tags=call_tags)
        return await self.oracle.run(question, tags=tags)

    async def aclose(self):
        self.closed = True


@register_pipeline("test_llm_echo")
def build_llm_echo(ctx):
    return LLMEchoPipeline(ctx)


def llm_factory(backend, cache_dir):
    return lambda trace_path: LLMClient(backend, cache_dir=cache_dir, trace_path=trace_path)


def test_llm_costs_cache_hits_and_judge_are_attributed(tmp_dir):
    backend = ScriptedBackend()
    factory = llm_factory(backend, tmp_dir / "cache")
    cfg = config(factory="test_llm_echo", judge="llm")
    pipeline_cost = pricing.cost_usd("claude-haiku-5-5", PIPELINE_USAGE)
    judge_cost = pricing.cost_usd("claude-sonnet-5-5", JUDGE_USAGE)

    first = run(cfg, tmp_dir, llm_factory=factory)
    rows = {r["id"]: r for r in read_jsonl(first.run_dir / "results.jsonl")}
    f001 = rows["f001"]  # olgusal: one pipeline call + one text judge call
    assert f001["cost_usd"] == pytest.approx(pipeline_cost)
    assert f001["judge_cost_usd"] == pytest.approx(judge_cost)
    assert (f001["llm_calls"], f001["judge_llm_calls"], f001["cached_llm_calls"]) == (1, 1, 0)
    assert f001["input_tokens"] == 1000 and f001["latency_eligible"] is True
    assert f001["judge_verdict"] == {"reason": "ok", "correct": True}
    assert rows["f002"]["judge_llm_calls"] == 0  # numeric: no judge
    assert rows["f006"]["metrics"]["key_point_coverage"] == 1.0

    op = first.summary["overall"]["operational"]
    assert op["cost_usd_total"] == pytest.approx(11 * pipeline_cost)
    assert op["judge_cost_usd_total"] == pytest.approx(3 * judge_cost)  # f001, f005, f006
    assert op["latency_n"] == 11
    assert first.summary["judge"]["llm_calls"] == 3

    traces = {t["qid"]: t for t in read_jsonl(first.run_dir / "traces.jsonl")}
    calls = traces["f001"]["llm_calls"]
    assert [c["role"] for c in calls] == ["pipeline", "judge"]
    assert calls[0]["tags"]["step"] == "generate" and calls[0]["tags"]["qid"] == "f001"
    assert calls[1]["model"] == "claude-sonnet-5-5"
    assert traces["f001"]["retrieval"][0]["doc"]
    assert traces["f001"]["question"] == "Akbank'ın 2025 sonunda kaç şubesi var?"

    # same prompts again: every call is a cache hit
    calls_before = backend.calls
    second = run(cfg, tmp_dir, llm_factory=factory)
    assert backend.calls == calls_before
    rows2 = {r["id"]: r for r in read_jsonl(second.run_dir / "results.jsonl")}
    assert rows2["f001"]["cost_usd"] == pytest.approx(pipeline_cost)  # original cost
    assert rows2["f001"]["cached_llm_calls"] == 1
    assert rows2["f001"]["latency_eligible"] is False
    op2 = second.summary["overall"]["operational"]
    assert op2["cost_usd_total"] == pytest.approx(op["cost_usd_total"])
    assert (op2["latency_n"], op2["latency_excluded_cached"]) == (0, 11)
    assert op2["latency_ms_p50"] is None


def test_non_llm_exception_is_isolated_and_judge_errors_are_kept(tmp_dir):
    backend = ScriptedBackend(judge_reply="not json")
    cfg = config(factory="test_llm_echo", params={"raise_on": "f003"}, judge="llm")
    result = run(cfg, tmp_dir, llm_factory=llm_factory(backend, None))
    rows = {r["id"]: r for r in read_jsonl(result.run_dir / "results.jsonl")}
    assert rows["f003"]["status"] == "error"
    assert rows["f003"]["error"] == "KeyError: 'not an LLM error'"
    assert rows["f004"]["status"] == "ok"
    f001 = rows["f001"]
    assert f001["status"] == "ok"
    assert f001["judge_error"].startswith("LLMCallError: ")
    assert f001["metrics"]["text_correct"] is None
    assert f001["metrics"]["answer_score"] is None
    assert result.summary["judge"]["errors"] == 2  # f001, f005 (text); f006 key points ok
    assert result.summary["overall"]["operational"]["judge_errors"] == 2
    assert result.summary["overall"]["metrics"]["answer_accuracy"]["n"] == 9


def test_untagged_calls_are_reported_as_unattributed(tmp_dir):
    backend = ScriptedBackend()
    cfg = config(factory="test_llm_echo", params={"untagged": True})
    result = run(cfg, tmp_dir, llm_factory=llm_factory(backend, None), limit=2)
    assert result.summary["unattributed_llm_calls"]["n"] == 2
    assert result.summary["overall"]["operational"]["cost_usd_total"] == 0
