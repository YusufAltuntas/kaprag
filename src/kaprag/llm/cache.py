"""Disk cache of successful LLM responses, so re-running a configuration spends no quota."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

from kaprag.llm.types import Usage


def cache_key(
    *, backend: str, model: str, system: str, user: str, schema: dict[str, Any] | None
) -> str:
    """SHA-256 over every input that can change the response."""
    payload = json.dumps(
        {"backend": backend, "model": model, "system": system, "user": user, "schema": schema},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class DiskCache:
    """One JSON file per key under `root/<key[:2]>/<key>.json`."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        """Return the stored entry (`text`, `usage`, `latency_ms`) or None."""
        path = self._path(key)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return None
        data["usage"] = Usage(**data["usage"])
        return data

    def put(self, key: str, *, text: str, usage: Usage, latency_ms: float) -> None:
        """Atomically store a successful response."""
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        entry = {"text": text, "usage": asdict(usage), "latency_ms": latency_ms}
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(entry, f, ensure_ascii=False)
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
