"""Run the eval: `uv run python scripts/run_eval.py --config <name> --split dev|test|pilot`.

Writes runs/YYYY-MM-DD-<topic>-<split>/ (README.md, config.yaml, results.jsonl,
summary.json, traces.jsonl). See `kaprag.eval.cli` for the options.
"""

import sys

from kaprag.eval.cli import main

if __name__ == "__main__":
    sys.exit(main())
