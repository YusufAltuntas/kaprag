"""Run the eval: `uv run python scripts/run_eval.py --config <name> --split dev|test|pilot`.

Writes runs/YYYY-MM-DD-<topic>-<split>/ (README.md, config.yaml, results.jsonl,
summary.json, traces.jsonl). See `kaprag.eval.cli` for the options. Local settings
(`DATABASE_URL`, `LLM_BACKEND`, ...) are read from `.env` without overriding the
environment.
"""

import sys

from dotenv import load_dotenv

from kaprag.corpus import REPO_ROOT
from kaprag.eval.cli import main

if __name__ == "__main__":
    load_dotenv(REPO_ROOT / ".env", override=False)
    sys.exit(main())
