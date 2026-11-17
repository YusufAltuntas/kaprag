"""Refuse to overwrite a gold questions file that may hold the user's review work.

The drafting CLI rewrites its `--out` file in full. Overwriting is allowed only with an
explicit flag, and never when the file holds a verified or human-authored question:
those exist only through the review app and would be lost.
"""

from __future__ import annotations

import json
from pathlib import Path


def overwrite_refusal(path: Path, *, overwrite: bool) -> str | None:
    """Why writing drafts to `path` must be refused, or None when it is safe."""
    path = Path(path)
    if not path.exists():
        return None
    if not overwrite:
        return f"{path} exists; pass --overwrite to replace it"
    protected: list[str] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            return f"{path}:{lineno} is not valid JSON; refusing to overwrite it"
        if not isinstance(row, dict):
            return f"{path}:{lineno} is not a question object; refusing to overwrite it"
        if row.get("verified") is True or row.get("author") == "human":
            protected.append(str(row.get("id", f"line {lineno}")))
    if protected:
        shown = ", ".join(protected[:5]) + (" ..." if len(protected) > 5 else "")
        return (
            f"{path} holds {len(protected)} verified or human-authored question(s) "
            f"({shown}); refusing to overwrite it even with --overwrite"
        )
    return None
