"""`scripts/run_eval.py` command line: test-split guards, end-to-end fake run, errors."""

import json
from pathlib import Path

from kaprag.eval.cli import main

FIXTURE = Path(__file__).parent / "fixtures" / "eval" / "questions.jsonl"


def args(tmp_dir, *extra):
    return [
        "--config", "fake-oracle", "--questions", str(FIXTURE),
        "--runs-dir", str(tmp_dir / "runs"), *extra,
    ]  # fmt: skip


def test_test_split_refused_without_confirmation(tmp_dir, capsys):
    assert main(args(tmp_dir, "--split", "test")) == 2
    err = capsys.readouterr().err
    assert "refusing to run the test split" in err
    assert "--confirm-test" in err
    assert not (tmp_dir / "runs").exists()


def test_test_split_runs_with_confirmation(tmp_dir):
    assert main(args(tmp_dir, "--split", "test", "--confirm-test")) == 0
    [run_dir] = (tmp_dir / "runs").iterdir()
    assert run_dir.name.endswith("-fake-oracle-test")
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["overall"]["n"] == 2


def test_end_to_end_dev_run(tmp_dir):
    assert main(args(tmp_dir, "--split", "dev", "--concurrency", "2")) == 0
    [run_dir] = (tmp_dir / "runs").iterdir()
    assert {p.name for p in run_dir.iterdir()} == {
        "README.md", "config.yaml", "results.jsonl", "summary.json", "traces.jsonl"
    }  # fmt: skip
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["run"]["config_name"] == "fake-oracle"
    assert summary["overall"]["n"] == 11


def test_pilot_with_limit(tmp_dir):
    assert main(args(tmp_dir, "--split", "pilot", "--limit", "3")) == 0
    [run_dir] = (tmp_dir / "runs").iterdir()
    assert len((run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()) == 3


def test_pilot_refused_when_it_selects_test_split_questions(tmp_dir, capsys):
    # the fixture's pilot set includes f011, a `split: test` question
    assert main(args(tmp_dir, "--split", "pilot")) == 2
    err = capsys.readouterr().err
    assert "f011" in err and "--confirm-test" in err
    assert not (tmp_dir / "runs").exists()


def test_pilot_with_test_split_questions_runs_with_confirmation(tmp_dir):
    assert main(args(tmp_dir, "--split", "pilot", "--confirm-test")) == 0
    [run_dir] = (tmp_dir / "runs").iterdir()
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["overall"]["n"] == 7


def test_missing_question_file_and_config(tmp_dir, capsys):
    missing = ["--config", "fake-oracle", "--split", "dev", "--questions", str(tmp_dir / "x")]
    assert main(missing + ["--runs-dir", str(tmp_dir / "runs")]) == 1
    assert "question file not found" in capsys.readouterr().err
    assert main(["--config", "no-such-config", "--split", "dev", "--questions", str(FIXTURE)]) == 1
    assert "not found" in capsys.readouterr().err
    assert not (tmp_dir / "runs").exists()
