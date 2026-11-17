"""Gold question schema (eval/README.md §3) and JSONL load/save helpers.

Field names and enumerations follow the spec exactly. Every field is required (no
defaults), unknown fields are rejected, and nothing here ever sets `verified`: it is
read and written back verbatim. `evidence[].page` is the 1-based PDF page index
(printed page numbers are not used anywhere in the project) and may be null.

Cross-field rules enforced (all stated or implied by eval/README.md §1, §3, §4):
- `type == "cevapsiz"`  <=>  `expected_behavior == "abstain"`.
- `expected_behavior == "clarify"`  =>  `type == "belirsiz"`.
- `expected_behavior == "answer_with_assumption"`  =>  `basis` is set (the basis the
  answer must state; the behaviour check compares against it).
- `type == "sentez"`  =>  `key_points` is non-empty.
- `expected.value` set  =>  `expected.unit` and `expected.tolerance` set; money (`TL`)
  and counts (`adet`) are integers.
- `verified == true`  <=>  `verified_at` is set.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_QUESTIONS = REPO_ROOT / "eval" / "gold" / "questions.jsonl"

Bank = Literal["garanti", "akbank", "isbank", "yapikredi"]
Split = Literal["dev", "test"]
QuestionType = Literal[
    "olgusal", "tablo", "hesaplama", "karsilastirma", "sentez", "cevapsiz", "belirsiz"
]
Behavior = Literal["answer", "abstain", "clarify", "answer_with_assumption"]
Basis = Literal["konsolide", "solo"]
Unit = Literal["TL", "%", "adet"]
Author = Literal["llm", "human"]

QUESTION_TYPES: tuple[str, ...] = QuestionType.__args__  # type: ignore[attr-defined]
BEHAVIORS: tuple[str, ...] = Behavior.__args__  # type: ignore[attr-defined]

# Years a question may mention. Wider than the corpus (2023-2025): reports carry
# prior-year comparatives and 5-year summaries, and `cevapsiz` questions may ask
# about years outside the corpus.
YEAR_MIN, YEAR_MAX = 2015, 2030
DOC_ID_PATTERN = r"^(garanti|akbank|isbank|yapikredi)-(2023|2024|2025)$"


class _Strict(BaseModel):
    # strict: "true" / 1 / "12" are not silently coerced into bool / int fields
    model_config = ConfigDict(extra="forbid", strict=True)


class Tolerance(_Strict):
    type: Literal["rel", "abs"]
    amount: int | float = Field(ge=0)  # int kept as int so files round-trip unchanged


class Expected(_Strict):
    answer: str
    value: int | float | None
    unit: Unit | None
    tolerance: Tolerance | None


class Evidence(_Strict):
    doc: str = Field(pattern=DOC_ID_PATTERN)
    page: int | None = Field(ge=1)  # 1-based PDF page index


class Question(_Strict):
    id: str = Field(min_length=1)
    split: Split
    pilot: bool
    type: QuestionType
    question: str = Field(min_length=1)
    banks: list[Bank]
    years: list[int]
    basis: Basis | None
    expected_behavior: Behavior
    expected: Expected
    evidence: list[Evidence]
    key_points: list[str]
    derived_from: list[str]
    author: Author
    llm_checked: bool
    verified: bool
    verified_at: str | None
    notes: str

    @model_validator(mode="after")
    def _cross_field_rules(self) -> Question:
        errors = []
        for year in self.years:
            if not YEAR_MIN <= year <= YEAR_MAX:
                errors.append(f"year {year} outside {YEAR_MIN}-{YEAR_MAX}")
        if (self.type == "cevapsiz") != (self.expected_behavior == "abstain"):
            errors.append("type 'cevapsiz' and expected_behavior 'abstain' go together")
        if self.expected_behavior == "clarify" and self.type != "belirsiz":
            errors.append("expected_behavior 'clarify' requires type 'belirsiz'")
        if self.expected_behavior == "answer_with_assumption" and self.basis is None:
            errors.append("expected_behavior 'answer_with_assumption' requires basis")
        if self.type == "sentez" and not self.key_points:
            errors.append("type 'sentez' requires key_points")
        exp = self.expected
        if exp.value is not None:
            if exp.unit is None or exp.tolerance is None:
                errors.append("expected.value requires expected.unit and expected.tolerance")
            if exp.unit in ("TL", "adet") and not float(exp.value).is_integer():
                errors.append(f"expected.value for unit {exp.unit!r} must be an integer")
        if self.verified != (self.verified_at is not None):
            errors.append("verified and verified_at must be set together")
        if errors:
            raise ValueError("; ".join(errors))
        return self

    @property
    def gold_pages(self) -> list[tuple[str, int]]:
        """Distinct (doc, page) evidence pairs with a page, in file order."""
        seen: dict[tuple[str, int], None] = {}
        for ev in self.evidence:
            if ev.page is not None:
                seen.setdefault((ev.doc, ev.page), None)
        return list(seen)


class GoldSetError(ValueError):
    """A gold file line is malformed; the message names the line and question id."""


def parse_question(line: str, *, where: str = "<string>") -> Question:
    """Validate one JSONL line; errors name `where` and the question id."""
    try:
        data = json.loads(line)
    except json.JSONDecodeError as exc:
        raise GoldSetError(f"{where}: invalid JSON: {exc}") from None
    qid = data.get("id") if isinstance(data, dict) else None
    try:
        return Question.model_validate(data)
    except ValidationError as exc:
        raise GoldSetError(f"{where}: question {qid!r}: {exc}") from None


def load_questions(path: Path = DEFAULT_QUESTIONS) -> list[Question]:
    """Read a gold JSONL file (blank lines skipped); fails on the first bad line or dup id."""
    path = Path(path)
    questions: list[Question] = []
    seen: set[str] = set()
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        q = parse_question(line, where=f"{path}:{lineno}")
        if q.id in seen:
            raise GoldSetError(f"{path}:{lineno}: duplicate question id {q.id!r}")
        seen.add(q.id)
        questions.append(q)
    return questions


def dump_question(question: Question) -> str:
    """One JSONL line, fields in schema order, non-ASCII kept as is."""
    return json.dumps(question.model_dump(mode="json"), ensure_ascii=False)


def save_questions(questions: Iterable[Question], path: Path = DEFAULT_QUESTIONS) -> None:
    """Write questions as JSONL (atomic replace), one line per question, in the given order."""
    path = Path(path)
    text = "".join(dump_question(q) + "\n" for q in questions)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    tmp.replace(path)
