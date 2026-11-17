"""The draft record every gold-drafting stage produces, and its conversion to a `Question`.

A `Draft` is a gold question without the fields assembly decides (`id`, `split`,
`pilot`) and with provenance the stages need: a stable `key` (ordering, ids and
`derived_from` references before ids exist), the `source` stage, and for figures read
from a page the `metric` / `quote` they came from. Authorship is fixed: every draft
becomes `author: "llm"`, `verified: false`, `verified_at: null`.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from kaprag.eval.schema import (
    Bank,
    Basis,
    Behavior,
    Evidence,
    Expected,
    Question,
    QuestionType,
    Split,
    Tolerance,
)

Source = Literal["page", "probe", "derived", "abstain", "ambiguous"]

BANK_NAMES: dict[str, str] = {
    "garanti": "Garanti BBVA",
    "akbank": "Akbank",
    "isbank": "İş Bankası",
    "yapikredi": "Yapı Kredi",
}
# Genitive forms ("Akbank'ın"), for questions composed in code.
BANK_GENITIVE: dict[str, str] = {
    "garanti": "Garanti BBVA'nın",
    "akbank": "Akbank'ın",
    "isbank": "İş Bankası'nın",
    "yapikredi": "Yapı Kredi'nin",
}
BANKS: tuple[str, ...] = tuple(BANK_NAMES)

# Tolerances by unit (eval/README.md §3) for figures read from a report.
TOLERANCE_BY_UNIT: dict[str, Tolerance] = {
    "TL": Tolerance(type="rel", amount=0.01),
    "%": Tolerance(type="abs", amount=0.2),
    "adet": Tolerance(type="abs", amount=0),
}
# Tolerance of a computed percentage change (the spec's own `hesaplama` example).
COMPUTED_PERCENT_TOLERANCE = Tolerance(type="abs", amount=0.5)


class Draft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    source: Source
    type: QuestionType
    question: str
    banks: list[Bank]
    years: list[int]
    basis: Basis | None
    expected_behavior: Behavior
    expected: Expected
    evidence: list[Evidence]
    key_points: list[str] = []
    derived_from: list[str] = []  # keys of source drafts
    llm_checked: bool = False
    notes: str = ""
    metric: str | None = None  # metric id (probes, YoY pairs)
    quote: str | None = None  # the figure(s) as written on the page


def to_question(
    draft: Draft, *, qid: str, split: Split, pilot: bool, key_to_id: Mapping[str, str]
) -> Question:
    """The gold `Question` for a draft; `derived_from` keys are mapped to ids."""
    return Question(
        id=qid,
        split=split,
        pilot=pilot,
        type=draft.type,
        question=draft.question,
        banks=list(draft.banks),
        years=list(draft.years),
        basis=draft.basis,
        expected_behavior=draft.expected_behavior,
        expected=draft.expected,
        evidence=list(draft.evidence),
        key_points=list(draft.key_points),
        derived_from=[key_to_id[k] for k in draft.derived_from],
        author="llm",
        llm_checked=draft.llm_checked,
        verified=False,
        verified_at=None,
        notes=draft.notes,
    )


def write_drafts(path: Path, drafts: Iterable[Draft]) -> int:
    """Write drafts as JSONL (atomic replace); returns the count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    lines = [json.dumps(d.model_dump(mode="json"), ensure_ascii=False) for d in drafts]
    tmp.write_text("".join(line + "\n" for line in lines), encoding="utf-8", newline="\n")
    tmp.replace(path)
    return len(lines)


def read_drafts(path: Path) -> Iterator[Draft]:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield Draft.model_validate_json(line)


def fold(text: str) -> str:
    """Lower-case with Turkish dotted/dotless I handled (for text checks)."""
    return text.replace("İ", "i").replace("I", "ı").lower()
