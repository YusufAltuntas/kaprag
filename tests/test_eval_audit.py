"""Applying agent-audit decisions to a gold file (kaprag.eval.audit, scripts/apply_audit.py)."""

import importlib.util
import json
from pathlib import Path

import pytest

from kaprag.eval.audit import (
    AuditError,
    AuditRefused,
    apply_audit_file,
    apply_decisions,
    format_summary,
    parse_decisions,
)
from kaprag.eval.review import rejected_path_for
from kaprag.eval.schema import Question, load_questions, save_questions

FIXTURE = Path(__file__).parent / "fixtures" / "eval" / "questions.jsonl"
SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "apply_audit.py"
BASE = json.loads(FIXTURE.read_text(encoding="utf-8").splitlines()[2])  # f003, llm, unverified
DAY = "2026-11-18"


def ev(doc: str, page: int | None) -> dict:
    return {"doc": doc, "page": page}


def make(qid: str, **overrides) -> Question:
    data = json.loads(json.dumps(BASE)) | {"id": qid}
    for key, value in overrides.items():
        if "." in key:
            outer, inner = key.split(".")
            data[outer][inner] = value
        else:
            data[key] = value
    return Question.model_validate(data)


def decision(qid: str, action: str, *, set=None, remove=None, reason="") -> dict:  # noqa: A002
    return {
        "id": qid,
        "audit_verdict": "ok" if action == "pass" else action,
        "action": action,
        "set": set or {},
        "remove_evidence": remove or [],
        "reason": reason,
    }


def decisions(*rows: dict) -> list:
    return parse_decisions({"rules": {}, "decisions": {r["id"]: r for r in rows}})


def gold_set() -> list[Question]:
    """q1..q3 single questions, q4 derived as the union of q1+q2, q5 derived but not a union."""
    return [
        make("q1", evidence=[ev("akbank-2024", 10), ev("akbank-2024", 11)]),
        make("q2", evidence=[ev("garanti-2024", 20)]),
        make("q3", evidence=[ev("isbank-2024", 30), ev("isbank-2024", 31)]),
        make(
            "q4",
            type="karsilastirma",
            derived_from=["q1", "q2"],
            evidence=[ev("akbank-2024", 10), ev("akbank-2024", 11), ev("garanti-2024", 20)],
        ),
        make(
            "q5",
            type="karsilastirma",
            derived_from=["q1", "q3"],
            evidence=[ev("akbank-2024", 10), ev("isbank-2024", 30)],
        ),
    ]


def by_id(questions):
    return {q.id: q for q in questions}


# --- decisions file ------------------------------------------------------------------


def test_parse_decisions_checks_shape():
    assert [d.id for d in decisions(decision("q1", "pass"))] == ["q1"]
    with pytest.raises(AuditError, match="unknown action"):
        decisions(decision("q1", "approve"))
    with pytest.raises(AuditError, match="not settable"):
        decisions(decision("q1", "fix", set={"question": "x"}))
    with pytest.raises(AuditError, match="not settable"):
        decisions(decision("q1", "fix", set={"verified": True}))
    with pytest.raises(AuditError, match="'pass' cannot set"):
        decisions(decision("q1", "pass", set={"expected.value": 1}))
    with pytest.raises(AuditError, match="'flag' cannot"):
        decisions(decision("q1", "flag", remove=[ev("akbank-2024", 11)]))
    with pytest.raises(AuditError, match="nothing to apply"):
        decisions(decision("q1", "fix"))
    with pytest.raises(AuditError, match="key 'q9'"):
        parse_decisions({"decisions": {"q9": decision("q1", "pass")}})


# --- per-action behaviour ------------------------------------------------------------


def test_pass_sets_audit_and_removes_listed_extra_pages():
    out = apply_decisions(
        gold_set(),
        decisions(decision("q1", "pass", remove=[ev("akbank-2024", 11)]), decision("q2", "pass")),
        date=DAY,
    )
    q = by_id(out.questions)
    assert [(e.doc, e.page) for e in q["q1"].evidence] == [("akbank-2024", 10)]
    assert q["q1"].audit.model_dump() == {
        "status": "passed",
        "by": "agent",
        "at": DAY,
        "note": "removed 1 unsupported evidence page(s)",
    }
    assert q["q2"].audit.note == ""
    assert q["q3"].audit is None  # no decision, untouched
    assert out.evidence_removed == 1
    assert out.counts["pass"] == 2


def test_first_evidence_entry_is_never_removed():
    with pytest.raises(AuditError, match="first evidence"):
        apply_decisions(
            gold_set(), decisions(decision("q1", "pass", remove=[ev("akbank-2024", 10)])), date=DAY
        )


def test_removing_a_page_that_is_not_there_is_an_error():
    with pytest.raises(AuditError, match="not in its evidence"):
        apply_decisions(
            gold_set(), decisions(decision("q1", "pass", remove=[ev("akbank-2024", 99)])), date=DAY
        )


def test_fix_applies_dotted_keys_and_evidence():
    tol = {"type": "abs", "amount": 1}
    reason = "The page gives 13,1; " + "x" * 400
    out = apply_decisions(
        gold_set(),
        decisions(
            decision(
                "q3",
                "fix",
                set={
                    "expected.value": 13.1,
                    "expected.answer": "Yüzde 13,1 arttı.",
                    "expected.tolerance": tol,
                    "evidence": [ev("isbank-2024", 31), ev("isbank-2024", 32)],
                },
                remove=[ev("isbank-2024", 32)],
                reason=reason,
            )
        ),
        date=DAY,
    )
    q = by_id(out.questions)["q3"]
    original = by_id(gold_set())["q3"]
    assert (q.expected.value, q.expected.answer) == (13.1, "Yüzde 13,1 arttı.")
    assert q.expected.tolerance.model_dump() == tol
    assert q.expected.unit == original.expected.unit  # untouched sibling field
    assert [(e.doc, e.page) for e in q.evidence] == [("isbank-2024", 31)]
    assert q.audit.status == "fixed"
    assert q.audit.note.startswith("The page gives 13,1;") and len(q.audit.note) <= 200
    protected = (
        "id", "split", "pilot", "verified", "verified_at", "author", "llm_checked",
        "question", "derived_from", "type", "banks", "years", "basis", "notes",
    )  # fmt: skip
    for field in protected:
        assert getattr(q, field) == getattr(original, field), field


def test_fix_result_must_validate():
    with pytest.raises(AuditError, match="q3"):
        apply_decisions(
            gold_set(),
            decisions(decision("q3", "fix", set={"expected.tolerance": None})),
            date=DAY,
        )


def test_flag_changes_only_the_audit():
    out = apply_decisions(
        gold_set(), decisions(decision("q2", "flag", reason="ambiguous wording")), date=DAY
    )
    q = by_id(out.questions)["q2"]
    assert q.audit.model_dump() == {
        "status": "flagged",
        "by": "agent",
        "at": DAY,
        "note": "ambiguous wording",
    }
    assert q.model_copy(update={"audit": None}) == by_id(gold_set())["q2"]
    assert not q.audited


def test_reject_moves_row_with_reason_in_notes():
    out = apply_decisions(
        gold_set(), decisions(decision("q3", "reject", reason="duplicate of q1")), date=DAY
    )
    assert "q3" not in by_id(out.questions)
    [moved] = out.rejected
    assert moved.id == "q3" and moved.audit is None
    assert moved.notes.endswith("Red gerekçesi: duplicate of q1")
    assert [q.id for q in out.questions] == ["q1", "q2", "q4", "q5"]


def test_unknown_decision_id_is_an_error():
    with pytest.raises(AuditError, match="q9"):
        apply_decisions(gold_set(), decisions(decision("q9", "pass")), date=DAY)


# --- refusals ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "override",
    [{"verified": True, "verified_at": "2026-11-17T00:00:00+00:00"}, {"author": "human"}],
)
def test_refuses_rows_verified_or_written_by_a_human(override):
    questions = gold_set()
    questions[1] = Question.model_validate(questions[1].model_dump(mode="json") | override)
    with pytest.raises(AuditRefused, match="q2"):
        apply_decisions(questions, decisions(decision("q2", "pass")), date=DAY)


def test_refuses_derived_recompute_of_a_verified_row():
    questions = gold_set()
    stamp = {"verified": True, "verified_at": "2026-11-17T00:00:00+00:00"}
    questions[3] = Question.model_validate(questions[3].model_dump(mode="json") | stamp)
    with pytest.raises(AuditRefused, match="q4"):
        apply_decisions(
            questions,
            decisions(decision("q1", "pass", remove=[ev("akbank-2024", 11)])),
            date=DAY,
        )


def test_refuses_rows_already_audited_unless_forced():
    once = apply_decisions(gold_set(), decisions(decision("q2", "pass")), date=DAY).questions
    with pytest.raises(AuditRefused, match="already"):
        apply_decisions(once, decisions(decision("q2", "pass")), date=DAY)
    again = apply_decisions(once, decisions(decision("q2", "pass")), date=DAY, force=True)
    assert again.questions == once


# --- derived evidence ----------------------------------------------------------------


def test_union_derived_evidence_is_recomputed_from_final_sources():
    out = apply_decisions(
        gold_set(),
        decisions(
            decision("q1", "pass", remove=[ev("akbank-2024", 11)]),
            decision("q3", "pass", remove=[ev("isbank-2024", 31)]),
        ),
        date=DAY,
    )
    q = by_id(out.questions)
    # q4 was exactly the union of q1+q2 -> follows its sources
    q4_pages = [(e.doc, e.page) for e in q["q4"].evidence]
    assert q4_pages == [("akbank-2024", 10), ("garanti-2024", 20)]
    # q5 was not a union -> left alone
    assert q["q5"].evidence == by_id(gold_set())["q5"].evidence
    assert out.derived_recomputed == ["q4"]
    assert q["q4"].audit is None  # recomputing evidence is not an audit verdict


def test_derived_evidence_left_alone_when_a_source_is_rejected():
    out = apply_decisions(
        gold_set(),
        decisions(
            decision("q1", "pass", remove=[ev("akbank-2024", 11)]),
            decision("q2", "reject", reason="dup"),
        ),
        date=DAY,
    )
    assert by_id(out.questions)["q4"].evidence == by_id(gold_set())["q4"].evidence
    assert out.derived_recomputed == []


def interleaved_set() -> list[Question]:
    """Derived rows laid out like `gold.evidence.propagate`: own page first, then sources."""
    return [
        make("q1", evidence=[ev("akbank-2024", 10), ev("akbank-2024", 11)]),
        make("q2", evidence=[ev("garanti-2024", 20), ev("garanti-2024", 21)]),
        make(
            "q4",
            type="karsilastirma",
            derived_from=["q1", "q2"],
            # set-equal to the union of q1+q2, but not in its order
            evidence=[
                ev("garanti-2024", 20),
                ev("akbank-2024", 10),
                ev("garanti-2024", 21),
                ev("akbank-2024", 11),
            ],
        ),
        make(
            "q5",
            type="karsilastirma",
            derived_from=["q1", "q2"],
            evidence=[ev("garanti-2024", 20), ev("akbank-2024", 10)],  # a subset: not a union
        ),
        make(
            "q6",
            type="karsilastirma",
            derived_from=["q1", "q2"],
            evidence=[
                ev("akbank-2024", 11),
                ev("garanti-2024", 20),
                ev("akbank-2024", 10),
                ev("garanti-2024", 21),
            ],  # fmt: skip
        ),
    ]


def test_set_union_derived_row_keeps_its_order_drops_removed_and_appends_new_pages():
    out = apply_decisions(
        interleaved_set(),
        decisions(
            decision("q1", "pass", remove=[ev("akbank-2024", 11)]),
            decision(
                "q2",
                "fix",
                set={"evidence": [ev("garanti-2024", 20), ev("garanti-2024", 22)]},
                reason="p21 does not state it; p22 does",
            ),
            decision("q6", "flag", reason="needs a human"),
        ),
        date=DAY,
    )
    q = by_id(out.questions)
    pages = [(e.doc, e.page) for e in q["q4"].evidence]
    assert pages == [("garanti-2024", 20), ("akbank-2024", 10), ("garanti-2024", 22)]
    assert q["q5"].evidence == by_id(interleaved_set())["q5"].evidence  # not a union
    assert q["q6"].evidence == by_id(interleaved_set())["q6"].evidence  # flagged: unchanged
    assert q["q6"].audit.status == "flagged"
    assert out.derived_recomputed == ["q4"]


# --- idempotence and file level ------------------------------------------------------

REAL_SHAPE = [
    decision("q1", "pass", remove=[ev("akbank-2024", 11)]),
    decision("q2", "flag", reason="needs a human"),
    decision("q3", "reject", reason="duplicate"),
    decision("q5", "fix", set={"expected.value": 14.0}, reason="recomputed"),
]


def write_inputs(tmp_dir: Path) -> tuple[Path, Path]:
    gold = tmp_dir / "gold" / "questions.jsonl"
    gold.parent.mkdir()
    save_questions(gold_set(), gold)
    dec = tmp_dir / "decisions.json"
    dec.write_text(
        json.dumps({"rules": {}, "decisions": {d["id"]: d for d in REAL_SHAPE}}), encoding="utf-8"
    )
    return gold, dec


def test_file_level_apply_and_forced_rerun_is_idempotent(tmp_dir):
    gold, dec = write_inputs(tmp_dir)
    out = apply_audit_file(gold, dec, date=DAY)
    assert [q.id for q in load_questions(gold)] == ["q1", "q2", "q4", "q5"]
    assert [q.id for q in load_questions(rejected_path_for(gold))] == ["q3"]
    assert out.counts == {"pass": 1, "fix": 1, "flag": 1, "reject": 1}
    first = (gold.read_bytes(), rejected_path_for(gold).read_bytes())

    with pytest.raises(AuditRefused):
        apply_audit_file(gold, dec, date=DAY)
    again = apply_audit_file(gold, dec, date=DAY, force=True)
    assert again.already_rejected == ["q3"]
    assert (gold.read_bytes(), rejected_path_for(gold).read_bytes()) == first


def test_dry_run_writes_nothing(tmp_dir):
    gold, dec = write_inputs(tmp_dir)
    before = gold.read_bytes()
    out = apply_audit_file(gold, dec, date=DAY, dry_run=True)
    assert out.counts["reject"] == 1
    assert gold.read_bytes() == before
    assert not rejected_path_for(gold).exists()


def test_no_path_sets_verified(tmp_dir):
    gold, dec = write_inputs(tmp_dir)
    apply_audit_file(gold, dec, date=DAY)
    rows = [json.loads(line) for line in gold.read_text(encoding="utf-8").splitlines()]
    assert all(r["verified"] is False and r["verified_at"] is None for r in rows)


def test_summary_mentions_every_count(tmp_dir):
    gold, dec = write_inputs(tmp_dir)
    text = format_summary(apply_audit_file(gold, dec, date=DAY, dry_run=True))
    needles = (
        "pass: 1", "fix: 1", "flag: 1", "reject: 1", "evidence pages removed: 1",
        "derived evidence recomputed: 1", "moved to rejected: 1", "hesaplama",
    )  # fmt: skip
    for needle in needles:
        assert needle in text, needle


# --- CLI ------------------------------------------------------------------------------


def load_script():
    spec = importlib.util.spec_from_file_location("apply_audit_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_dry_run_and_refusal_exit_codes(tmp_dir, capsys):
    gold, dec = write_inputs(tmp_dir)
    main = load_script().main
    before = gold.read_bytes()
    args = ["--decisions", str(dec), "--questions", str(gold), "--date", DAY]
    assert main([*args, "--dry-run"]) == 0
    assert "dry run" in capsys.readouterr().out
    assert gold.read_bytes() == before
    assert main(args) == 0
    capsys.readouterr()
    after = gold.read_bytes()
    assert main(args) == 2  # second run refused
    assert "already" in capsys.readouterr().err
    assert gold.read_bytes() == after
    assert main([*args, "--date", "18.11.2026"]) == 2


# --- translated rows (questions_en.jsonl) ----------------------------------------------


def translation(tid: str, source: Question) -> Question:
    data = source.model_dump(mode="json") | {
        "id": tid,
        "question": f"English wording of {source.id}?",
        "derived_from": [source.id],
        "notes": f"translation of {source.id}",
    }
    data["expected"] = data["expected"] | {"answer": "English reference answer."}
    return Question.model_validate(data)


def write_translations(tmp_dir: Path, sources=("q1", "q2", "q5")) -> Path:
    gold = by_id(gold_set())
    path = tmp_dir / "gold" / "questions_en.jsonl"
    save_questions([translation(f"en{s[1:]}", gold[s]) for s in sources], path)
    return path


def test_sync_translations_copies_final_evidence_and_expected_numbers(tmp_dir):
    gold, dec = write_inputs(tmp_dir)  # q1 loses a page, q5 gets a new value
    en = write_translations(tmp_dir)
    before = {q.id: q for q in load_questions(en)}
    out = apply_audit_file(gold, dec, date=DAY, translations_path=en)
    assert out.translations_synced == ["en1", "en5"]
    after = {q.id: q for q in load_questions(en)}
    final = by_id(load_questions(gold))
    for tid, sid in (("en1", "q1"), ("en5", "q5")):
        t, s = after[tid], final[sid]
        assert t.evidence == s.evidence
        assert (t.expected.value, t.expected.unit, t.expected.tolerance) == (
            s.expected.value, s.expected.unit, s.expected.tolerance
        )  # fmt: skip
        b = before[tid]
        assert (t.question, t.expected.answer, t.key_points, t.notes) == (
            b.question, b.expected.answer, b.key_points, b.notes
        )  # fmt: skip
        assert (t.id, t.split, t.verified, t.verified_at, t.audit) == (
            b.id, b.split, b.verified, b.verified_at, b.audit
        )  # fmt: skip
    assert after["en2"] == before["en2"]  # q2 was only flagged: nothing to sync
    assert "translations synced: 2 (en1 <- q1, en5 <- q5)" in format_summary(out)


@pytest.mark.parametrize("source", ["q3", "q9"])  # q3 is rejected, q9 does not exist
def test_sync_translations_refuses_rejected_or_missing_source(tmp_dir, source):
    gold, dec = write_inputs(tmp_dir)
    en = write_translations(tmp_dir, sources=("q1",))
    rows = load_questions(en)
    orphan = Question.model_validate(
        rows[0].model_dump(mode="json") | {"id": "enX", "derived_from": [source]}
    )
    save_questions([*rows, orphan], en)
    files = {p: p.read_bytes() for p in (gold, en)}
    with pytest.raises(AuditRefused, match="enX"):
        apply_audit_file(gold, dec, date=DAY, translations_path=en)
    assert {p: p.read_bytes() for p in files} == files
    assert not rejected_path_for(gold).exists()


def test_cli_sync_translations_refusal_exit_code(tmp_dir, capsys):
    gold, dec = write_inputs(tmp_dir)
    en = write_translations(tmp_dir, sources=("q1", "q3"))  # q3 is being rejected
    main = load_script().main
    args = ["--decisions", str(dec), "--questions", str(gold), "--date", DAY]
    assert main([*args, "--sync-translations", str(en)]) == 2
    assert "en3" in capsys.readouterr().err
    en = write_translations(tmp_dir, sources=("q1",))
    assert main([*args, "--sync-translations", str(en)]) == 0
    assert "translations synced: 1 (en1 <- q1)" in capsys.readouterr().out
