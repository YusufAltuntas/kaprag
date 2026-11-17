"""The English question set (eval/gold/questions_en.jsonl): hand translations of dev-split
questions of the main gold file, for the cross-lingual retrieval probe."""

from pathlib import Path

from kaprag.eval.schema import DEFAULT_QUESTIONS, load_questions

EN_QUESTIONS = Path(__file__).resolve().parents[1] / "eval" / "gold" / "questions_en.jsonl"


def test_english_questions_are_unverified_translations_of_dev_questions():
    english = load_questions(EN_QUESTIONS)  # validates every row against the schema
    main = {q.id: q for q in load_questions(DEFAULT_QUESTIONS)}
    assert 12 <= len(english) <= 20
    assert not {q.id for q in english} & main.keys()
    assert {q.type for q in english} >= {
        "olgusal", "tablo", "hesaplama", "karsilastirma", "sentez", "cevapsiz", "belirsiz"
    }  # fmt: skip
    for q in english:
        assert q.id.startswith("en")
        assert (q.split, q.pilot, q.author, q.verified, q.verified_at) == (
            "dev",
            False,
            "llm",
            False,
            None,
        )
        [source_id] = q.derived_from
        source = main[source_id]
        assert source.split == "dev", q.id
        assert q.question != source.question
        assert source_id in q.notes
        same = ("type", "banks", "years", "basis", "expected_behavior", "evidence")
        for field in same:
            assert getattr(q, field) == getattr(source, field), (q.id, field)
        assert (q.expected.value, q.expected.unit, q.expected.tolerance) == (
            source.expected.value, source.expected.unit, source.expected.tolerance
        )  # fmt: skip
        assert len(q.key_points) == len(source.key_points), q.id
