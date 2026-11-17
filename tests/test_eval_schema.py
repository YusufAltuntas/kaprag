"""Gold question schema validation and JSONL round-trip."""

import json
from pathlib import Path

import pytest

from kaprag.eval.schema import (
    GoldSetError,
    Question,
    load_questions,
    parse_question,
    save_questions,
)

FIXTURE = Path(__file__).parent / "fixtures" / "eval" / "questions.jsonl"


def base() -> dict:
    """A valid `tablo` question as a plain dict."""
    return json.loads(FIXTURE.read_text(encoding="utf-8").splitlines()[1])


def line(data: dict) -> str:
    return json.dumps(data, ensure_ascii=False)


def test_fixture_loads_and_covers_every_type_and_behavior():
    questions = load_questions(FIXTURE)
    assert len(questions) == 13
    assert {q.type for q in questions} == {
        "olgusal", "tablo", "hesaplama", "karsilastirma", "sentez", "cevapsiz", "belirsiz"
    }  # fmt: skip
    assert {q.expected_behavior for q in questions} == {
        "answer", "abstain", "clarify", "answer_with_assumption"
    }  # fmt: skip


def test_round_trip_is_byte_identical(tmp_dir):
    out = tmp_dir / "questions.jsonl"
    save_questions(load_questions(FIXTURE), out)
    assert out.read_bytes() == FIXTURE.read_bytes()


def test_round_trip_keeps_field_order_and_values():
    q = parse_question(line(base()))
    assert list(json.loads(json.dumps(q.model_dump(mode="json")))) == list(base())
    assert q.expected.value == 75_300_000_000
    assert isinstance(q.expected.value, int)


def test_save_never_changes_verified(tmp_dir):
    questions = load_questions(FIXTURE)
    out = tmp_dir / "q.jsonl"
    save_questions(questions, out)
    assert [q.verified for q in load_questions(out)] == [q.verified for q in questions]


def test_gold_pages_skip_null_and_deduplicate():
    data = base()
    data["evidence"] = [
        {"doc": "isbank-2024", "page": 120},
        {"doc": "isbank-2024", "page": None},
        {"doc": "isbank-2024", "page": 120},
        {"doc": "isbank-2023", "page": 7},
    ]
    assert parse_question(line(data)).gold_pages == [("isbank-2024", 120), ("isbank-2023", 7)]


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("type",), "tablolar"),
        (("split",), "train"),
        (("expected_behavior",), "refuse"),
        (("banks",), ["ziraat"]),
        (("basis",), "bireysel"),
        (("author",), "agent"),
        (("expected", "unit"), "USD"),
        (("expected", "tolerance"), {"type": "pct", "amount": 0.01}),
        (("expected", "tolerance"), {"type": "rel", "amount": -0.01}),
        (("evidence",), [{"doc": "ziraat-2024", "page": 3}]),
        (("evidence",), [{"doc": "akbank-2022", "page": 3}]),
        (("evidence",), [{"doc": "akbank-2024", "page": 0}]),
        (("years",), [1999]),
        (("verified",), "true"),
        (("pilot",), 1),
    ],
)
def test_unknown_values_fail_with_question_id(path, value):
    data = base()
    target = data
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(GoldSetError, match="'f002'"):
        parse_question(line(data))


def test_unknown_and_missing_fields_fail():
    data = base() | {"extra": 1}
    with pytest.raises(GoldSetError, match="extra"):
        parse_question(line(data))
    data = base()
    del data["notes"]
    with pytest.raises(GoldSetError, match="notes"):
        parse_question(line(data))


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"type": "cevapsiz"}, "cevapsiz"),
        ({"expected_behavior": "abstain"}, "cevapsiz"),
        ({"expected_behavior": "clarify"}, "clarify"),
        ({"type": "belirsiz", "expected_behavior": "answer_with_assumption", "basis": None},
         "requires basis"),
        ({"type": "sentez", "key_points": []}, "key_points"),
        ({"verified": False}, "verified_at"),
        ({"verified_at": None}, "verified_at"),
    ],
)  # fmt: skip
def test_cross_field_rules(changes, message):
    data = base() | changes
    with pytest.raises(GoldSetError, match=message):
        parse_question(line(data))


def test_value_rules():
    data = base()
    data["expected"]["unit"] = None
    with pytest.raises(GoldSetError, match="expected.unit"):
        parse_question(line(data))
    data = base()
    data["expected"]["value"] = 75.5
    with pytest.raises(GoldSetError, match="integer"):
        parse_question(line(data))


def test_load_reports_line_number_and_duplicates(tmp_dir):
    path = tmp_dir / "q.jsonl"
    path.write_text(line(base()) + "\n\n" + line(base()) + "\n", encoding="utf-8")
    with pytest.raises(GoldSetError, match=r"q\.jsonl:3: duplicate question id 'f002'"):
        load_questions(path)
    path.write_text(line(base()) + "\n{not json\n", encoding="utf-8")
    with pytest.raises(GoldSetError, match=r"q\.jsonl:2: invalid JSON"):
        load_questions(path)


def test_question_model_has_spec_fields_only():
    assert list(Question.model_fields) == [
        "id", "split", "pilot", "type", "question", "banks", "years", "basis",
        "expected_behavior", "expected", "evidence", "key_points", "derived_from",
        "author", "llm_checked", "verified", "verified_at", "audit", "notes",
    ]  # fmt: skip


# --- audit (agent audit, not human verification) -------------------------------------

AUDIT = {"status": "fixed", "by": "agent", "at": "2026-11-18", "note": "value corrected"}


def test_row_without_audit_key_loads_as_null_and_saves_it_after_verified_at(tmp_dir):
    data = base()
    del data["audit"]
    q = parse_question(line(data))
    assert q.audit is None
    out = tmp_dir / "q.jsonl"
    save_questions([q], out)
    keys = list(json.loads(out.read_text(encoding="utf-8")))
    assert keys[keys.index("verified_at") + 1] == "audit"
    assert json.loads(out.read_text(encoding="utf-8"))["audit"] is None


def test_audit_object_round_trips(tmp_dir):
    q = parse_question(line(base() | {"audit": AUDIT}))
    assert (q.audit.status, q.audit.by, q.audit.at) == ("fixed", "agent", "2026-11-18")
    assert q.audited
    assert not parse_question(line(base() | {"audit": AUDIT | {"status": "flagged"}})).audited
    assert not parse_question(line(base())).audited
    out = tmp_dir / "q.jsonl"
    save_questions([q], out)
    assert json.loads(out.read_text(encoding="utf-8"))["audit"] == AUDIT
    assert load_questions(out) == [q]


@pytest.mark.parametrize(
    "audit",
    [
        AUDIT | {"status": "verified"},
        AUDIT | {"status": "rejected"},
        AUDIT | {"by": "human"},
        AUDIT | {"at": "18.11.2026"},
        AUDIT | {"at": "2026-11-18T03:00:00"},
        AUDIT | {"note": None},
        AUDIT | {"extra": 1},
        {k: v for k, v in AUDIT.items() if k != "note"},
        "passed",
        True,
    ],
)
def test_invalid_audit_objects_fail(audit):
    with pytest.raises(GoldSetError, match="'f002'"):
        parse_question(line(base() | {"audit": audit}))
