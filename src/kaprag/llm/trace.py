"""JSONL trace of every LLM call; the source of truth for cost and latency metrics."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any


class TraceWriter:
    """Appends one JSON object per line; safe under concurrent calls."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def write(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line)
