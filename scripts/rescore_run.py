"""Re-score a finished run's stored outputs against the current gold file.

    uv run python scripts/rescore_run.py --run runs/2026-11-17-rerank-dev
    uv run python scripts/rescore_run.py --run runs/<folder> --questions <gold> --runs-dir <dir>

Writes runs/<original-name>-rescored[-N]/ (README.md, config.yaml, results.jsonl,
summary.json); the original folder is never modified. No pipeline runs: judge calls
only, through kaprag.llm and its cache. See `kaprag.eval.rescore`. Local settings
(`LLM_BACKEND`, ...) are read from `.env` without overriding the environment.

Exit codes: 0 re-scored, 1 the run cannot be re-scored.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

from kaprag.corpus import REPO_ROOT
from kaprag.eval.cli import summary_line
from kaprag.eval.config import ConfigError
from kaprag.eval.rescore import RescoreError, format_report, rescore_run
from kaprag.eval.runner import DEFAULT_RUNS_DIR
from kaprag.eval.schema import DEFAULT_QUESTIONS, GoldSetError

log = logging.getLogger("rescore_run")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Re-score a run against the current gold set.")
    parser.add_argument("--run", type=Path, required=True, help="original run folder")
    parser.add_argument(
        "--questions",
        type=Path,
        default=DEFAULT_QUESTIONS,
        help="gold JSONL file (default: eval/gold/questions.jsonl)",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS_DIR,
        help="parent of the new folder (default: runs/)",
    )
    return parser


def _fmt(group: dict, metric: str) -> str:
    value = group["metrics"][metric]["value"]
    return "n/a" if value is None else f"{value:.3f}"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        if not args.questions.is_file():
            raise GoldSetError(f"question file not found: {args.questions}")
        result = rescore_run(args.run, questions_path=args.questions, runs_dir=args.runs_dir)
    except (RescoreError, ConfigError, GoldSetError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except Exception:
        log.exception("re-scoring failed")
        return 1
    for line in format_report(result):
        print(line)
    s = result.summary
    print(summary_line(s))
    print(
        "page_recall@5: "
        + "; ".join(f"{g} {_fmt(s[g], 'page_recall@5')}" for g in ("overall", "audited"))
    )
    return 0


if __name__ == "__main__":
    load_dotenv(REPO_ROOT / ".env", override=False)
    sys.exit(main())
