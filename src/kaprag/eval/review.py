"""Write-back logic of the gold set review app (scripts/review_app.py), free of Streamlit.

Every write re-reads the file, applies one change to the question identified by id
and replaces the file atomically through the schema helpers, so all other rows stay
byte-for-byte as they were. A change is refused (`StaleQuestionError`) when the row
on disk differs from the row the user was shown.

`verified` is only ever set by `approve`, by `edit(..., approve=True)` and by
`add_human_question` (a question the user writes from the PDF page in the review app
is verified by its author). Each of these runs only on an explicit user action.

Rejected questions are moved to `rejected.jsonl` next to the questions file, so the
gold file only holds live questions and nothing is lost.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from kaprag.eval.schema import QUESTION_TYPES, Question, load_questions, save_questions
from kaprag.eval.split import assign_split_for_new

REJECTED_NAME = "rejected.jsonl"

# Fields the user may change from the edit and add forms. Everything else (id, split,
# pilot, derived_from, author, llm_checked, verified, verified_at) is kept from disk
# or set by the function itself.
EDITABLE_FIELDS: tuple[str, ...] = (
    "question",
    "type",
    "banks",
    "years",
    "basis",
    "expected_behavior",
    "expected",
    "evidence",
    "key_points",
    "notes",
)

_ID_PATTERN = re.compile(r"^q(\d+)$")


class ReviewError(ValueError):
    """A change cannot be applied (invalid values or a forbidden field)."""


class StaleQuestionError(ReviewError):
    """The question on disk is not the one the user was shown; reload before changing it."""


def rejected_path_for(questions_path: Path) -> Path:
    """The rejected-questions file that belongs to a questions file."""
    return Path(questions_path).with_name(REJECTED_NAME)


def load_for_review(path: Path) -> list[Question]:
    """Questions in file order; a missing file is an empty set."""
    path = Path(path)
    return load_questions(path) if path.exists() else []


def progress(questions: Iterable[Question]) -> dict[str, Any]:
    """Verified / total counts overall (`total`) and per question type (`by_type`)."""
    questions = list(questions)
    total = Counter(q.type for q in questions)
    done = Counter(q.type for q in questions if q.verified)
    return {
        "total": (sum(done.values()), len(questions)),
        "by_type": {t: (done[t], total[t]) for t in QUESTION_TYPES},
    }


def _now_iso(now: datetime | None) -> str:
    stamp = now or datetime.now(UTC)
    return stamp.astimezone(UTC).replace(microsecond=0).isoformat()


def _validate(data: dict[str, Any]) -> Question:
    try:
        return Question.model_validate(data)
    except ValidationError as exc:
        raise ReviewError(_format_errors(exc)) from None


def _format_errors(exc: ValidationError) -> str:
    parts = []
    for err in exc.errors():
        where = ".".join(str(p) for p in err["loc"])
        parts.append(f"{where}: {err['msg']}" if where else err["msg"])
    return "; ".join(parts)


def _check_editable(changes: dict[str, Any]) -> None:
    forbidden = sorted(set(changes) - set(EDITABLE_FIELDS))
    if forbidden:
        raise ReviewError(f"fields not editable: {', '.join(forbidden)}")


def _locate(questions: list[Question], shown: Question) -> int:
    for i, q in enumerate(questions):
        if q.id == shown.id:
            if q != shown:
                raise StaleQuestionError(f"question {shown.id} changed on disk")
            return i
    raise StaleQuestionError(f"question {shown.id} is no longer in the file")


def _replace(path: Path, shown: Question, new: Question) -> Question:
    questions = load_for_review(path)
    questions[_locate(questions, shown)] = new
    save_questions(questions, path)
    return new


def approve(path: Path, shown: Question, *, now: datetime | None = None) -> Question:
    """Mark one question verified now; nothing else in the row changes."""
    new = shown.model_copy(update={"verified": True, "verified_at": _now_iso(now)})
    return _replace(Path(path), shown, new)


def edit(
    path: Path,
    shown: Question,
    changes: dict[str, Any],
    *,
    approve: bool = False,
    now: datetime | None = None,
) -> Question:
    """Apply user edits to one question; `verified` changes only with `approve=True`."""
    _check_editable(changes)
    data = shown.model_dump(mode="json") | changes
    if approve:
        data |= {"verified": True, "verified_at": _now_iso(now)}
    return _replace(Path(path), shown, _validate(data))


def reject(path: Path, shown: Question, reason: str = "") -> Question:
    """Move one question to the rejected file, with the reason appended to `notes`."""
    path = Path(path)
    questions = load_for_review(path)
    index = _locate(questions, shown)
    notes = shown.notes
    if reason.strip():
        note = f"Red gerekçesi: {reason.strip()}"
        notes = f"{notes} | {note}" if notes else note
    moved = shown.model_copy(update={"notes": notes})
    rejected_path = rejected_path_for(path)
    rejected = load_for_review(rejected_path)
    if any(q.id == shown.id for q in rejected):
        raise ReviewError(f"question {shown.id} is already in {rejected_path.name}")
    # Rejected file first: a crash in between leaves a duplicate, never a lost row.
    save_questions([*rejected, moved], rejected_path)
    del questions[index]
    save_questions(questions, path)
    return moved


def next_question_id(questions: Iterable[Question], rejected: Iterable[Question]) -> str:
    """Next free `qNNN` id after every `q<number>` id in both files (ids are never reused)."""
    numbers = [int(m.group(1)) for q in [*questions, *rejected] if (m := _ID_PATTERN.match(q.id))]
    return f"q{max(numbers, default=0) + 1:03d}"


def add_human_question(
    path: Path, fields: dict[str, Any], *, now: datetime | None = None
) -> Question:
    """Append a question written by the user from a PDF page; it is verified on creation."""
    path = Path(path)
    _check_editable(fields)
    questions = load_for_review(path)
    rejected = load_for_review(rejected_path_for(path))
    data = {
        "id": next_question_id(questions, rejected),
        "split": "dev",  # placeholder, replaced by the split rule below
        "pilot": False,
        **fields,
        "derived_from": [],
        "author": "human",
        "llm_checked": False,
        "verified": True,
        "verified_at": _now_iso(now),
    }
    draft = _validate(data)
    new = draft.model_copy(update={"split": assign_split_for_new(draft, questions)})
    save_questions([*questions, new], path)
    return new


# --- form parsing (messages are shown in the Turkish UI) ----------------------------


def parse_years(text: str) -> list[int]:
    """Comma or space separated years."""
    try:
        return [int(part) for part in re.split(r"[,\s]+", text.strip()) if part]
    except ValueError:
        raise ReviewError(f"Yıllar sayı olmalı: {text!r}") from None


def parse_number(text: str) -> int | float | None:
    """Empty -> None; integers stay int; a decimal comma is accepted."""
    text = text.strip().replace(" ", "").replace("_", "")
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text.replace(",", "."))
    except ValueError:
        raise ReviewError(f"Sayı değil: {text!r}") from None


def parse_evidence(text: str) -> list[dict[str, Any]]:
    """One `<doc> <page>` per line; `-` for an unknown page."""
    evidence = []
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 2:
            raise ReviewError(f"Kanıt satırı '<doküman> <sayfa>' olmalı: {line!r}")
        doc, page = parts
        if page == "-":
            evidence.append({"doc": doc, "page": None})
        elif page.isdigit():
            evidence.append({"doc": doc, "page": int(page)})
        else:
            raise ReviewError(f"Kanıt sayfası sayı ya da '-' olmalı: {line!r}")
    return evidence


def format_evidence(question: Question) -> str:
    """Inverse of `parse_evidence` for pre-filling the edit form."""
    return "\n".join(f"{ev.doc} {'-' if ev.page is None else ev.page}" for ev in question.evidence)
