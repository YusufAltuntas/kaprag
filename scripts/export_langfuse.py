"""Export a finished run to Langfuse: `scripts/export_langfuse.py --run runs/<folder>`.

`--dry-run` builds every trace, observation and score and prints the counts without
the SDK, credentials or network. A real export needs the `observability` dependency
group and `LANGFUSE_HOST`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` (read from
`.env` without overriding the environment). See `kaprag.observability.export_cli`.
"""

import sys

from dotenv import load_dotenv

from kaprag.corpus import REPO_ROOT
from kaprag.observability.export_cli import main

if __name__ == "__main__":
    load_dotenv(REPO_ROOT / ".env", override=False)
    sys.exit(main())
