"""Validate a gold questions file against the schema and print its distribution report.

Run:
  uv run python scripts/gold_stats.py
  uv run python scripts/gold_stats.py --questions runs/2026-11-17-gold-draft-smoke/questions.jsonl

Exits 1 if the file does not load (schema error, duplicate id).
"""

import argparse
import sys
from pathlib import Path

from kaprag.eval.schema import DEFAULT_QUESTIONS, GoldSetError, load_questions
from kaprag.gold.stats import distribution_report


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    args = parser.parse_args()
    try:
        questions = load_questions(args.questions)
    except (GoldSetError, OSError) as exc:
        print(f"invalid gold file: {exc}", file=sys.stderr)
        return 1
    print(f"{args.questions}: schema ok")
    print(distribution_report(questions))
    return 0


if __name__ == "__main__":
    sys.exit(main())
