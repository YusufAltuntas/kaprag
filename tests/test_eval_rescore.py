"""Re-scoring a finished run's stored outputs against the current gold file."""

import hashlib
import importlib.util
import json
import re
from datetime import date
from pathlib import Path

import pytest
import yaml

from kaprag.eval.config import EvalConfig
from kaprag.eval.fake_pipeline import FakeOraclePipeline
from kaprag.eval.judge import KEY_POINTS_SYSTEM, TEXT_SYSTEM
from kaprag.eval.pipeline import register_pipeline
from kaprag.eval.rescore import RescoreError, rescore_run
from kaprag.eval.runner import run_eval
from kaprag.eval.schema import Question, load_questions, save_questions
from kaprag.llm import LLMClient, Usage
from kaprag.llm.types import BackendResponse

FIXTURE = Path(__file__).parent / "fixtures" / "eval" / "questions.jsonl"
SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "rescore_run.py"
DAY = date(2026, 11, 17)
RESCORE_DAY = date(2026, 11, 18)


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class JudgeBackend:
    """Judge prompts get a 'correct' / all-covered verdict; anything else gets text."""

    name = "fake"

    def __init__(self):
        self.calls = 0

    async def generate(self, system, user, model):
        self.calls += 1
        if system == TEXT_SYSTEM:
            text = '{"reason": "ok", "correct": true}'
        elif system == KEY_POINTS_SYSTEM:
            n = int(re.search(r"KEY POINTS \((\d+)\)", user).group(1))
            text = json.dumps({"reason": "ok", "covered": [True] * n})
        else:
            text = "draft"
        return BackendResponse(text=text, usage=Usage(input_tokens=100, output_tokens=10))


class EchoLLMPipeline:
    """One tagged LLM call per question (so cost is non-zero), then an oracle answer."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.oracle = FakeOraclePipeline(ctx.gold, error_rate=0, fail_rate=0, seed=0)

    async def run(self, question, *, tags):
        if self.ctx.params.get("raise_on") == tags["qid"]:
            raise RuntimeError("boom")
        await self.ctx.llm().complete(
            "echo", question, "claude-haiku-5-5", tags={**tags, "step": "generate"}
        )
        return await self.oracle.run(question, tags=tags)


@register_pipeline("test_rescore_echo")
def build_echo(ctx):
    return EchoLLMPipeline(ctx)


@pytest.fixture
def env(tmp_dir):
    """An original run folder (dev split, LLM judge through a fake backend + disk cache)."""
    backend = JudgeBackend()
    cache = tmp_dir / "cache"

    def factory(trace_path):
        return LLMClient(backend, cache_dir=cache, trace_path=trace_path)

    cfg = EvalConfig(
        description="original run",
        topic="unit",
        pipeline={"factory": "test_rescore_echo", "params": {"raise_on": "f009"}},
        judge={"kind": "llm"},
        concurrency=3,
    )
    result = run_eval(
        cfg,
        config_name="unit",
        questions=load_questions(FIXTURE),
        questions_path=FIXTURE,
        split="dev",
        runs_dir=tmp_dir / "runs",
        today=DAY,
        llm_factory=factory,
    )
    return {"backend": backend, "factory": factory, "run_dir": result.run_dir, "tmp": tmp_dir}


def folder_hashes(path: Path) -> dict[str, str]:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(path.iterdir())}


def rescore(env, questions_path, **kwargs):
    return rescore_run(
        env["run_dir"],
        questions_path=questions_path,
        runs_dir=env["tmp"] / "rescored",
        llm_factory=env["factory"],
        today=RESCORE_DAY,
        **kwargs,
    )


def edited_gold(tmp_dir: Path, edits: dict[str, dict], *, drop=(), add=()) -> Path:
    questions = []
    for q in load_questions(FIXTURE):
        if q.id in drop:
            continue
        data = q.model_dump(mode="json")
        for key, value in edits.get(q.id, {}).items():
            outer, _, inner = key.partition(".")
            if inner:
                data[outer][inner] = value
            else:
                data[outer] = value
        questions.append(Question.model_validate(data))
    questions += list(add)
    path = tmp_dir / "gold" / "questions.jsonl"
    path.parent.mkdir(exist_ok=True)
    save_questions(questions, path)
    return path


# --- identity -------------------------------------------------------------------------


def test_unchanged_gold_reproduces_the_original_exactly(env):
    before = folder_hashes(env["run_dir"])
    calls_before = env["backend"].calls
    result = rescore(env, FIXTURE)
    original = json.loads((env["run_dir"] / "summary.json").read_text(encoding="utf-8"))
    for group in ("overall", "verified", "by_type"):
        assert result.summary[group] == original[group], group
    assert read_jsonl(result.run_dir / "results.jsonl") == read_jsonl(
        env["run_dir"] / "results.jsonl"
    )
    assert result.summary["judge"] == original["judge"] | {"llm_calls_fresh": 0,
                                                          "llm_calls_cached": 3}  # fmt: skip
    assert env["backend"].calls == calls_before  # every judge call was a cache hit
    assert folder_hashes(env["run_dir"]) == before  # original folder untouched


def test_new_folder_files_and_metadata(env):
    result = rescore(env, FIXTURE)
    assert result.run_dir.name == "2026-11-17-unit-dev-rescored"
    assert result.run_dir.parent == env["tmp"] / "rescored"
    assert {p.name for p in result.run_dir.iterdir()} == {
        "README.md", "config.yaml", "results.jsonl", "summary.json"
    }  # fmt: skip
    readme = (result.run_dir / "README.md").read_text(encoding="utf-8")
    assert readme.count("\n") == 1 and "2026-11-17-unit-dev" in readme
    sha = hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
    assert sha[:12] in readme

    cfg = yaml.safe_load((result.run_dir / "config.yaml").read_text(encoding="utf-8"))
    original = yaml.safe_load((env["run_dir"] / "config.yaml").read_text(encoding="utf-8"))
    assert {k: v for k, v in cfg.items() if k not in ("rescore", "run")} == {
        k: v for k, v in original.items() if k != "run"
    }
    block = cfg["rescore"]
    assert block["original_run_id"] == "2026-11-17-unit-dev"
    assert block["original_questions_sha256"] == original["run"]["questions_sha256"]
    assert block["questions_sha256"] == sha
    assert block["date"] == "2026-11-18"
    assert block["n_original"] == block["n_scored"] == 11
    assert result.summary["rescore"] == block

    # the run block names the gold version actually scored against
    for run in (cfg["run"], result.summary["run"]):
        assert run["questions_sha256"] == sha
        assert run["n_questions"] == 11
        assert run["questions_path"] == "tests/fixtures/eval/questions.jsonl"
        assert run["run_id"] == original["run"]["run_id"]
    assert block["original_n_questions"] == original["run"]["n_questions"]
    assert block["original_questions_path"] == original["run"]["questions_path"]

    second = rescore(env, FIXTURE)
    assert second.run_dir.name == "2026-11-17-unit-dev-rescored-2"


# --- a changed gold file -----------------------------------------------------------


def test_changed_gold_rescores_stored_outputs(env):
    new_q = Question.model_validate(
        load_questions(FIXTURE)[0].model_dump(mode="json") | {"id": "f099"}
    )
    gold = edited_gold(
        env["tmp"],
        {
            "f002": {"expected.value": 80_000_000_000},  # numeric answer now wrong
            "f001": {"evidence": [{"doc": "akbank-2025", "page": 35}]},  # retrieval now misses
            "f005": {"expected.answer": "Garanti BBVA, yeni ifade."},  # one fresh judge call
            "f004": {"question": "Reworded question text?"},  # stored answer is stale
        },
        drop=("f003",),  # rejected
        add=(new_q,),
    )
    original_rows = {r["id"]: r for r in read_jsonl(env["run_dir"] / "results.jsonl")}
    before = folder_hashes(env["run_dir"])
    calls_before = env["backend"].calls
    result = rescore(env, gold)
    rows = {r["id"]: r for r in read_jsonl(result.run_dir / "results.jsonl")}

    block = result.summary["rescore"]
    assert block["dropped_not_in_gold"] == ["f003"]
    assert block["excluded_question_changed"] == ["f004"]
    assert block["missing_from_run"] == ["f099"]
    assert (block["n_original"], block["n_scored"]) == (11, 9)
    assert "f003" not in rows and "f004" not in rows and "f099" not in rows

    assert original_rows["f002"]["metrics"]["numeric_correct"] is True
    assert rows["f002"]["metrics"]["numeric_correct"] is False
    assert rows["f002"]["metrics"]["answer_score"] == 0.0
    assert original_rows["f001"]["metrics"]["page_recall@5"] == 1.0
    assert rows["f001"]["metrics"]["page_recall@5"] == 0.0

    assert env["backend"].calls == calls_before + 1
    assert (
        result.summary["judge"]["llm_calls_fresh"],
        result.summary["judge"]["llm_calls_cached"],
    ) == (1, 2)  # noqa: E501
    assert (block["judge_calls_fresh"], block["judge_calls_cached"]) == (1, 2)

    operational = ("latency_ms", "latency_eligible", "cost_usd", "input_tokens",
                   "output_tokens", "llm_calls", "cached_llm_calls", "status", "error",
                   "answer", "retrieved")  # fmt: skip
    for qid, row in rows.items():
        for key in operational:
            assert row[key] == original_rows[qid][key], (qid, key)
    assert rows["f009"]["status"] == "error"  # pipeline failure carried over, scored as 0
    assert rows["f009"]["metrics"]["answer_score"] == 0.0
    op = result.summary["overall"]["operational"]
    assert op["cost_usd_total"] == pytest.approx(sum(r["cost_usd"] for r in rows.values()))
    assert result.summary["overall"]["n"] == 9
    assert folder_hashes(env["run_dir"]) == before


def test_audit_status_flows_into_rows_and_audited_group(env):
    audit = {"status": "passed", "by": "agent", "at": "2026-11-18", "note": ""}
    gold = edited_gold(env["tmp"], {"f001": {"audit": audit}, "f002": {"audit": audit}})
    result = rescore(env, gold)
    rows = {r["id"]: r for r in read_jsonl(result.run_dir / "results.jsonl")}
    assert rows["f001"]["audit"] == "passed" and rows["f004"]["audit"] is None
    assert result.summary["audited"]["n"] == 2


# --- refusals --------------------------------------------------------------------------


def test_refuses_a_rescored_folder_and_a_folder_without_traces(env):
    first = rescore(env, FIXTURE)
    with pytest.raises(RescoreError, match="already a re-scoring"):
        rescore_run(first.run_dir, questions_path=FIXTURE, runs_dir=env["tmp"] / "x")
    copy = env["tmp"] / "copy"
    copy.mkdir()
    for name in ("config.yaml", "results.jsonl", "summary.json", "README.md"):
        (copy / name).write_bytes((env["run_dir"] / name).read_bytes())
    with pytest.raises(RescoreError, match="traces.jsonl"):
        rescore_run(copy, questions_path=FIXTURE, runs_dir=env["tmp"] / "x")
    assert not (env["tmp"] / "x").exists()


# --- CLI ---------------------------------------------------------------------------------


def load_script():
    spec = importlib.util.spec_from_file_location("rescore_run_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_on_a_test_split_run_with_fake_judge(tmp_dir, capsys):
    cfg = EvalConfig(
        description="fake test-split run",
        topic="unit",
        pipeline={"factory": "fake_oracle", "params": {"error_rate": 0, "fail_rate": 0}},
        judge={"kind": "fake"},
    )
    original = run_eval(
        cfg,
        config_name="unit",
        questions=load_questions(FIXTURE),
        questions_path=FIXTURE,
        split="test",
        runs_dir=tmp_dir / "runs",
        today=DAY,
    ).run_dir
    main = load_script().main
    args = ["--run", str(original), "--questions", str(FIXTURE), "--runs-dir", str(tmp_dir / "o")]
    assert main(args) == 0
    out = capsys.readouterr().out
    assert "re-scoring of stored outputs, not a new test run" in out
    assert "judge calls: fresh 0, cached 0" in out
    [new] = (tmp_dir / "o").iterdir()
    assert new.name == "2026-11-17-unit-test-rescored"
    assert main(["--run", str(tmp_dir / "nope"), "--questions", str(FIXTURE)]) == 1


def test_changed_gold_run_block_carries_current_hash_and_count(env):
    gold = edited_gold(env["tmp"], {}, drop=("f003",))
    result = rescore(env, gold)
    sha = hashlib.sha256(gold.read_bytes()).hexdigest()
    original = yaml.safe_load((env["run_dir"] / "config.yaml").read_text(encoding="utf-8"))
    run = result.summary["run"]
    assert (run["questions_sha256"], run["n_questions"]) == (sha, 10)
    block = result.summary["rescore"]
    assert block["original_questions_sha256"] == original["run"]["questions_sha256"] != sha
    assert block["original_n_questions"] == 11


def test_failed_rescore_leaves_no_folder(env):
    broken = env["tmp"] / "broken"
    broken.mkdir()
    for name in ("config.yaml", "summary.json", "traces.jsonl", "README.md"):
        (broken / name).write_bytes((env["run_dir"] / name).read_bytes())
    rows = read_jsonl(env["run_dir"] / "results.jsonl")
    rows[0]["answer"] = {"not": "an answer"}  # a stored output that cannot be scored
    (broken / "results.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    out = env["tmp"] / "out"
    with pytest.raises(ValueError):
        rescore_run(broken, questions_path=FIXTURE, runs_dir=out, llm_factory=env["factory"])
    assert not out.exists() or not any(out.iterdir())
