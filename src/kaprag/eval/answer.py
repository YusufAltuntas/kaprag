"""The system's structured answer (eval/README.md §5).

A pipeline returns this for every question; numeric scoring reads `value`/`unit`
deterministically, behaviour scoring reads `abstained`, `clarification` and `basis`.
It is also suitable as the `schema=` of the generation call.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Citation(BaseModel):
    doc: str  # doc id, e.g. "akbank-2024"
    page: int = Field(ge=1)  # 1-based PDF page index


class SystemAnswer(BaseModel):
    answer: str
    value: int | float | None = None  # in `unit`; a scale word in `unit` is applied
    unit: str | None = None  # e.g. "TL", "milyar TL", "%", "adet"
    basis: Literal["konsolide", "solo"] | None = None
    citations: list[Citation] = Field(default_factory=list)
    abstained: bool = False
    clarification: str | None = None
