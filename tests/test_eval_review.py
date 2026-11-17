"""Write-back logic of the review app (kaprag.eval.review): approve, edit, reject, add."""

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import pytest

from kaprag.eval.review import (
    EDITABLE_FIELDS,
    ReviewError,
    StaleQuestionError,
    add_human_question,
    approve,
    edit,
    load_for_review,
    next_question_id,
    parse_evidence,
    parse_number,
    parse_years,
    progress,
    reject,
    rejected_path_for,
)
from kaprag.eval.schema import Question, load_questions, save_questions

FIXTURE = Path(__file__).parent / "fixtures" / "eval" / "questions.jsonl"
NOW = datetime(2026, 11, 18, 3, 4, 5, tzinfo=UTC)
NOW_ISO = "2026-11-18T03:04:05+00:00"


@pytest.fixture
def gold(tmp_dir):
    """A writable copy of the fixture gold file."""
    path = tmp_dir / "gold" / "questions.jsonl"
    path.parent.mkdir()
    shutil.copyfile(FIXTURE, path)
    return path


def by_id(path: Path) -> dict[str, Question]:
    return {q.id: q for q in load_questions(path)}


def lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def changes_of(q: Question) -> dict:
    data = q.model_dump(mode="json")
    return {k: data[k] for k in EDITABLE_FIELDS}


def human_draft(**overrides) -> dict:
    data = {
        "question": "Garanti'nin 2024 sonunda kaç ATM'si vardı?",
        "type": "olgusal",
        "banks": ["garanti"],
        "years": [2024],
        "basis": None,
        "expected_behavior": "answer",
        "expected": {
            "answer": "5.400 ATM",
            "value": 5400,
            "unit": "adet",
            "tolerance": {"type": "abs", "amount": 0},
        },
        "evidence": [{"doc": "garanti-2024", "page": 15}],
        "key_points": [],
        "notes": "",
    }
    data.update(overrides)
    return data


# --- loading -------------------------------------------------------------------------


def test_missing_file_is_an_empty_set(tmp_dir):
    assert load_for_review(tmp_dir / "nope" / "questions.jsonl") == []


def test_rejected_file_sits_next_to_questions(tmp_dir):
    assert rejected_path_for(tmp_dir / "x" / "q.jsonl") == tmp_dir / "x" / "rejected.jsonl"


def test_load_and_save_without_action_changes_nothing(gold, tmp_dir):
    before = gold.read_bytes()
    questions = load_for_review(gold)
    out = tmp_dir / "copy.jsonl"
    save_questions(questions, out)
    assert out.read_bytes() == before
    assert [(q.verified, q.verified_at) for q in load_questions(out)] == [
        (q.verified, q.verified_at) for q in load_questions(FIXTURE)
    ]


def test_progress_counts_verified_per_type(gold):
    p = progress(load_for_review(gold))
    assert p["total"] == (4, 13)
    assert p["by_type"]["tablo"] == (1, 3)
    assert p["by_type"]["cevapsiz"] == (1, 2)


# --- approve -------------------------------------------------------------------------


def test_approve_changes_exactly_one_row(gold):
    before = lines(gold)
    shown = by_id(gold)["f003"]
    result = approve(gold, shown, now=NOW)
    after = lines(gold)
    assert result.verified is True and result.verified_at == NOW_ISO
    changed = [i for i, (a, b) in enumerate(zip(before, after, strict=True)) if a != b]
    assert changed == [2]
    row = json.loads(after[2])
    assert (row["verified"], row["verified_at"]) == (True, NOW_ISO)
    expected = json.loads(before[2]) | {"verified": True, "verified_at": NOW_ISO}
    assert row == expected


def test_approve_default_timestamp_is_utc_now(gold):
    result = approve(gold, by_id(gold)["f003"])
    stamp = datetime.fromisoformat(result.verified_at)
    assert stamp.utcoffset().total_seconds() == 0
    assert abs((datetime.now(UTC) - stamp).total_seconds()) < 60


def test_approve_refuses_a_row_changed_on_disk(gold):
    shown = by_id(gold)["f003"]
    questions = load_questions(gold)
    questions[2] = questions[2].model_copy(update={"notes": "changed elsewhere"})
    save_questions(questions, gold)
    disk = gold.read_bytes()
    with pytest.raises(StaleQuestionError):
        approve(gold, shown, now=NOW)
    assert gold.read_bytes() == disk


def test_approve_refuses_a_row_gone_from_disk(gold):
    shown = by_id(gold)["f003"]
    save_questions([q for q in load_questions(gold) if q.id != "f003"], gold)
    with pytest.raises(StaleQuestionError):
        approve(gold, shown, now=NOW)


# --- edit ----------------------------------------------------------------------------


def test_edit_valid_changes_only_that_row_and_keeps_unverified(gold):
    before = lines(gold)
    shown = by_id(gold)["f003"]
    changes = changes_of(shown)
    changes["question"] = "Yapı Kredi'nin net kârı 2024'ten 2025'e ne kadar değişti?"
    changes["evidence"] = [{"doc": "yapikredi-2025", "page": 41}]
    result = edit(gold, shown, changes)
    after = lines(gold)
    assert [i for i, (a, b) in enumerate(zip(before, after, strict=True)) if a != b] == [2]
    assert result.question.startswith("Yapı Kredi'nin net kârı")
    assert result.evidence[0].page == 41
    assert (result.verified, result.verified_at) == (False, None)
    assert by_id(gold)["f003"] == result


def test_edit_keeps_verified_state_of_verified_question(gold):
    shown = by_id(gold)["f002"]
    changes = changes_of(shown) | {"notes": "düzeltildi"}
    result = edit(gold, shown, changes, now=NOW)
    assert (result.verified, result.verified_at) == (True, shown.verified_at)


def test_edit_and_approve_sets_verified(gold):
    shown = by_id(gold)["f003"]
    result = edit(gold, shown, changes_of(shown) | {"notes": "ok"}, approve=True, now=NOW)
    assert (result.verified, result.verified_at) == (True, NOW_ISO)
    assert by_id(gold)["f003"].notes == "ok"


def test_edit_invalid_raises_and_writes_nothing(gold):
    before = gold.read_bytes()
    shown = by_id(gold)["f003"]
    bad = changes_of(shown) | {"type": "cevapsiz"}  # cevapsiz requires abstain
    with pytest.raises(ReviewError, match="abstain"):
        edit(gold, shown, bad)
    bad = changes_of(shown) | {"evidence": [{"doc": "ziraat-2024", "page": 3}]}
    with pytest.raises(ReviewError):
        edit(gold, shown, bad)
    assert gold.read_bytes() == before


def test_edit_cannot_touch_protected_fields(gold):
    shown = by_id(gold)["f003"]
    for field in ("verified", "verified_at", "id", "split", "author"):
        with pytest.raises(ReviewError, match=field):
            edit(gold, shown, changes_of(shown) | {field: None})


def test_edit_refuses_stale_row(gold):
    shown = by_id(gold)["f003"]
    approve(gold, shown, now=NOW)
    on_disk = gold.read_bytes()
    with pytest.raises(StaleQuestionError):
        edit(gold, shown, changes_of(shown) | {"notes": "x"})
    assert gold.read_bytes() == on_disk


# --- reject --------------------------------------------------------------------------


def test_reject_moves_row_to_rejected_file(gold):
    before = lines(gold)
    shown = by_id(gold)["f004"]
    reject(gold, shown, reason="Sayfada bu oran yok")
    after = lines(gold)
    assert after == before[:3] + before[4:]
    rejected = load_questions(rejected_path_for(gold))
    assert [q.id for q in rejected] == ["f004"]
    assert rejected[0].model_copy(update={"notes": shown.notes}) == shown
    assert "Sayfada bu oran yok" in rejected[0].notes
    assert shown.notes in rejected[0].notes

    reject(gold, by_id(gold)["f006"])
    rejected = load_questions(rejected_path_for(gold))
    assert [q.id for q in rejected] == ["f004", "f006"]
    assert rejected[1].notes == by_id(FIXTURE)["f006"].notes


def test_reject_refuses_stale_row(gold):
    shown = by_id(gold)["f004"]
    edit(gold, shown, changes_of(shown) | {"notes": "x"})
    with pytest.raises(StaleQuestionError):
        reject(gold, shown)
    assert "f004" in by_id(gold)
    assert not rejected_path_for(gold).exists()


# --- add a human question -----------------------------------------------------------


def test_next_id_skips_ids_in_both_files():
    def q(qid):
        return Question.model_validate(json.loads(lines(FIXTURE)[0]) | {"id": qid})

    assert next_question_id([], []) == "q001"
    assert next_question_id([q("q007"), q("f099")], [q("q012")]) == "q013"
    assert next_question_id([q("q999")], []) == "q1000"


def test_add_human_question_fields(gold):
    before = lines(gold)
    result = add_human_question(gold, human_draft(), now=NOW)
    after = lines(gold)
    assert after[:-1] == before
    assert result.id == "q001"
    assert result.author == "human"
    assert result.llm_checked is False
    assert result.pilot is False
    assert result.derived_from == []
    assert (result.verified, result.verified_at) == (True, NOW_ISO)
    assert result.split in ("dev", "test")
    assert by_id(gold)["q001"] == result


def test_add_human_question_never_reuses_a_rejected_id(gold):
    first = add_human_question(gold, human_draft(), now=NOW)
    reject(gold, first)
    second = add_human_question(gold, human_draft(), now=NOW)
    assert (first.id, second.id) == ("q001", "q002")


def test_add_human_question_creates_missing_file(tmp_dir):
    path = tmp_dir / "new" / "questions.jsonl"
    result = add_human_question(path, human_draft(), now=NOW)
    assert [q.id for q in load_questions(path)] == [result.id]


def test_add_human_question_invalid_writes_nothing(gold):
    before = gold.read_bytes()
    with pytest.raises(ReviewError):
        add_human_question(gold, human_draft(type="sentez"), now=NOW)  # needs key_points
    with pytest.raises(ReviewError, match="verified"):
        add_human_question(gold, human_draft(verified=False), now=NOW)
    assert gold.read_bytes() == before


def test_add_human_question_uses_split_rule(gold):
    from kaprag.eval.split import assign_split_for_new

    existing = load_questions(gold)
    result = add_human_question(gold, human_draft(), now=NOW)
    placeholder = result.model_copy(update={"split": "dev"})
    assert result.split == assign_split_for_new(placeholder, existing)


# --- form parsing helpers -----------------------------------------------------------


def test_parse_years():
    assert parse_years("2024, 2025") == [2024, 2025]
    assert parse_years("  ") == []
    with pytest.raises(ReviewError):
        parse_years("2024, yirmi")


def test_parse_number():
    assert parse_number("") is None
    assert parse_number("75300000000") == 75_300_000_000
    assert isinstance(parse_number("700"), int)
    assert parse_number("12.5") == 12.5
    assert parse_number("12,5") == 12.5
    with pytest.raises(ReviewError):
        parse_number("on iki")


def test_parse_evidence():
    assert parse_evidence("akbank-2024 87\n\nisbank-2023 -\n") == [
        {"doc": "akbank-2024", "page": 87},
        {"doc": "isbank-2023", "page": None},
    ]
    with pytest.raises(ReviewError):
        parse_evidence("akbank-2024 sayfa")


# --- audit field (agent audit; never touched by review flows) -------------------------

AUDIT = {"status": "fixed", "by": "agent", "at": "2026-11-18", "note": "value corrected"}


@pytest.fixture
def audited_gold(gold):
    """The fixture copy with an agent audit on f003 (unverified) and f002 (verified)."""
    questions = load_questions(gold)
    for i, q in enumerate(questions):
        if q.id in ("f002", "f003"):
            questions[i] = Question.model_validate(q.model_dump(mode="json") | {"audit": AUDIT})
    save_questions(questions, gold)
    return gold


def test_approve_keeps_audit(audited_gold):
    result = approve(audited_gold, by_id(audited_gold)["f003"], now=NOW)
    assert result.audit.model_dump() == AUDIT
    assert by_id(audited_gold)["f003"].audit.model_dump() == AUDIT


def test_edit_keeps_audit_and_cannot_change_it(audited_gold):
    shown = by_id(audited_gold)["f003"]
    result = edit(audited_gold, shown, changes_of(shown) | {"notes": "x"}, approve=True, now=NOW)
    assert result.audit.model_dump() == AUDIT
    assert by_id(audited_gold)["f003"].audit.model_dump() == AUDIT
    shown = by_id(audited_gold)["f002"]
    with pytest.raises(ReviewError, match="audit"):
        edit(audited_gold, shown, changes_of(shown) | {"audit": None})


def test_reject_keeps_audit(audited_gold):
    reject(audited_gold, by_id(audited_gold)["f003"], reason="r")
    [moved] = load_questions(rejected_path_for(audited_gold))
    assert moved.audit.model_dump() == AUDIT


def test_human_question_has_no_audit(gold):
    result = add_human_question(gold, human_draft(), now=NOW)
    assert result.audit is None
    assert json.loads(lines(gold)[-1])["audit"] is None
