"""Command line for the eval runner (`scripts/run_eval.py`).

    uv run python scripts/run_eval.py --config fake-oracle --split dev
    uv run python scripts/run_eval.py --config baseline --split pilot --limit 5
    uv run python scripts/run_eval.py --config final --split test --confirm-test

Exit codes: 0 run completed (whatever the scores), 1 run could not be completed,
2 usage error or the test split refused (also a `--split pilot` selection holding
`split: test` questions without `--confirm-test`).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from kaprag.eval.config import ConfigError, load_config
from kaprag.eval.runner import DEFAULT_RUNS_DIR, SPLITS, run_eval, select_questions
from kaprag.eval.schema import DEFAULT_QUESTIONS, GoldSetError, load_questions

log = logging.getLogger("run_eval")

TEST_SPLIT_REFUSAL = (
    "refusing to run the test split: it is reserved for important versions (the baseline "
    "and the final candidate, eval/README.md section 2); ablations use --split dev. "
    "Pass --confirm-test to run it anyway."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the kaprag eval on a gold question split.")
    parser.add_argument("--config", required=True, help="config name: configs/<name>.yaml")
    parser.add_argument("--split", required=True, choices=SPLITS)
    parser.add_argument(
        "--questions",
        type=Path,
        default=DEFAULT_QUESTIONS,
        help="gold JSONL file (default: eval/gold/questions.jsonl)",
    )
    parser.add_argument("--limit", type=int, default=None, help="first N selected questions")
    parser.add_argument(
        "--concurrency", type=int, default=None, help="override the config's concurrency"
    )
    parser.add_argument(
        "--confirm-test",
        action="store_true",
        help="required with --split test, and with --split pilot if it selects test questions",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS_DIR,
        help="parent of the run folder (default: runs/)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.split == "test" and not args.confirm_test:
        print(TEST_SPLIT_REFUSAL, file=sys.stderr)
        return 2
    if args.limit is not None and args.limit < 1:
        print("--limit must be >= 1", file=sys.stderr)
        return 2
    if args.concurrency is not None and args.concurrency < 1:
        print("--concurrency must be >= 1", file=sys.stderr)
        return 2

    try:
        config = load_config(args.config)
        if args.concurrency is not None:
            config = config.model_copy(update={"concurrency": args.concurrency})
        if not args.questions.is_file():
            raise GoldSetError(f"question file not found: {args.questions}")
        questions = load_questions(args.questions)
        if args.split == "pilot" and not args.confirm_test:
            selected = select_questions(questions, "pilot", args.limit)
            if test_ids := [q.id for q in selected if q.split == "test"]:
                print(
                    f"refusing to run the pilot: it selects {len(test_ids)} test-split "
                    f"question(s) {test_ids}, reserved for important versions (eval/README.md "
                    "section 2). Pass --confirm-test to run it anyway.",
                    file=sys.stderr,
                )
                return 2
        result = run_eval(
            config,
            config_name=args.config,
            questions=questions,
            questions_path=args.questions,
            split=args.split,
            runs_dir=args.runs_dir,
            limit=args.limit,
        )
    except (ConfigError, GoldSetError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except Exception:
        log.exception("eval run failed")
        return 1

    log.info("run folder: %s", result.run_dir)
    log.info("%s", summary_line(result.summary))
    return 0


def summary_line(summary: dict) -> str:
    """One console line: counts and answer accuracy overall, verified and agent-audited."""
    overall, verified, audited = summary["overall"], summary["verified"], summary["audited"]
    return (
        f"questions: {overall['n']} (ok {overall['n_ok']}, error {overall['n_error']}); "
        f"answer_accuracy: {_fmt(overall['metrics']['answer_accuracy']['value'])} "
        f"(verified n={verified['n']}: {_fmt(verified['metrics']['answer_accuracy']['value'])}; "
        f"agent-audited n={audited['n']}: "
        f"{_fmt(audited['metrics']['answer_accuracy']['value'])})"
    )


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"
