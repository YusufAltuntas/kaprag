"""Shared plumbing for the drafting stages: LLM calls that never abort the run, and the
drop / failure log written to the run folder.

A failed LLM call (`LLMCallError`, already traced by `kaprag.llm`) is recorded as a
failure and the item is skipped; a draft rejected by a code or LLM check is recorded
as a drop with its reason. Both go to the run folder (`failures.jsonl`, `drops.jsonl`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from kaprag.llm import LLMCallError

SONNET = "claude-sonnet-5-5"
HAIKU = "claude-haiku-5-5"


@dataclass
class RunLog:
    drops: list[dict] = field(default_factory=list)
    failures: list[dict] = field(default_factory=list)

    def drop(self, stage: str, key: str, reason: str, item: Any = None) -> None:
        if isinstance(item, BaseModel):
            item = item.model_dump(mode="json")
        self.drops.append({"stage": stage, "key": key, "reason": reason, "item": item})

    def fail(self, stage: str, key: str, error: str) -> None:
        self.failures.append({"stage": stage, "key": key, "error": error})

    def write(self, run_dir: Path) -> None:
        for name, rows in (("drops.jsonl", self.drops), ("failures.jsonl", self.failures)):
            text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
            (run_dir / name).write_text(text, encoding="utf-8", newline="\n")


async def call_llm(
    client: Any,
    log: RunLog,
    *,
    stage: str,
    key: str,
    system: str,
    user: str,
    schema: type[BaseModel],
    model: str = SONNET,
) -> Any | None:
    """Parsed reply of one schema call, or None (recorded as a failure) if it fails."""
    try:
        result = await client.complete(
            system, user, model, schema, tags={"stage": stage, "key": key}
        )
    except LLMCallError as exc:
        log.fail(stage, key, str(exc))
        return None
    return result.parsed
