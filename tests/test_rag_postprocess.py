"""RAG post-processing: cited source numbers -> {doc, page}; value/unit -> canonical units."""

import pytest

from kaprag.eval.answer import Citation, SystemAnswer
from kaprag.eval.pipeline import RetrievedItem
from kaprag.rag.generate import GeneratedAnswer
from kaprag.rag.postprocess import build_answer, map_citations, normalize_value

SOURCES = [
    RetrievedItem(chunk_id="a:1", doc="akbank-2024", page_start=87, page_end=88, score=0.9),
    RetrievedItem(chunk_id="a:2", doc="akbank-2024", page_start=88, page_end=88, score=0.8),
    RetrievedItem(chunk_id="g:1", doc="garanti-2025", page_start=50, page_end=50, score=0.7),
]


def gen(**kwargs):
    base = {"answer": "Cevap.", "citations": []}
    return GeneratedAnswer.model_validate({**base, **kwargs})


# --- citations -------------------------------------------------------------


def test_map_citations_expands_page_spans_and_deduplicates():
    citations, dropped = map_citations([1, 2, 3, 1], SOURCES)
    assert citations == [
        Citation(doc="akbank-2024", page=87),
        Citation(doc="akbank-2024", page=88),
        Citation(doc="garanti-2025", page=50),
    ]
    assert dropped == []


def test_map_citations_drops_numbers_outside_the_context():
    citations, dropped = map_citations([0, 3, 4, 9, -1], SOURCES)
    assert citations == [Citation(doc="garanti-2025", page=50)]
    assert dropped == [0, 4, 9, -1]


def test_map_citations_empty():
    assert map_citations([], SOURCES) == ([], [])
    assert map_citations([1], []) == ([], [1])


def test_build_answer_also_reads_markers_in_the_answer_text():
    answer, debug = build_answer(gen(answer="Kâr arttı [3] ve [1].", citations=[3]), SOURCES)
    assert answer.citations == [
        Citation(doc="garanti-2025", page=50),
        Citation(doc="akbank-2024", page=87),
        Citation(doc="akbank-2024", page=88),
    ]
    assert debug["cited_sources"] == [3, 1]
    assert debug["dropped_citations"] == []


def test_build_answer_records_dropped_citations():
    answer, debug = build_answer(gen(answer="Bkz. [7].", citations=[2, 5]), SOURCES)
    assert answer.citations == [Citation(doc="akbank-2024", page=88)]
    assert debug["dropped_citations"] == [5, 7]


# --- numbers ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "unit", "expected"),
    [
        ("45,2", "milyar TL", (45_200_000_000, "TL")),
        (45.2, "milyar TL", (45_200_000_000, "TL")),
        ("1.234.567", "bin TL", (1_234_567_000, "TL")),
        ("(1.234)", "Bin TL", (-1_234_000, "TL")),
        ("18,2", "%", (18.2, "%")),
        ("%18,2", None, (18.2, "%")),
        ("12,5", "yüzde", (12.5, "%")),
        ("700", "şube", (700, "adet")),
        (12900, "adet", (12900, "adet")),
        ("1,5", "trilyon TL", (1_500_000_000_000, "TL")),
        ("3.450", None, (3450, None)),
    ],
)
def test_normalize_value_applies_scale_words(value, unit, expected):
    (got_value, got_unit), error = normalize_value(value, unit)
    assert error is None
    assert (got_value, got_unit) == expected
    assert type(got_value) is type(expected[0])


@pytest.mark.parametrize(
    ("value", "unit"),
    [
        ("45,2", "milyon ABD doları"),
        ("yaklaşık kırk", "TL"),
        ("12-15", "%"),
        ("45,2", "milyar 2024 TL"),
    ],
)
def test_normalize_value_failure_returns_reason(value, unit):
    result, error = normalize_value(value, unit)
    assert result == (None, None)
    assert error


def test_normalize_value_without_value():
    assert normalize_value(None, "TL") == ((None, None), None)
    assert normalize_value("", "TL") == ((None, None), None)


def test_build_answer_normalises_numbers():
    answer, debug = build_answer(
        gen(
            answer="Konsolide net kâr 42,1 milyar TL [1].",
            value="42,1",
            unit="milyar TL",
            basis="konsolide",
            citations=[1],
        ),
        SOURCES,
    )
    assert answer == SystemAnswer(
        answer="Konsolide net kâr 42,1 milyar TL [1].",
        value=42_100_000_000,
        unit="TL",
        basis="konsolide",
        citations=[Citation(doc="akbank-2024", page=87), Citation(doc="akbank-2024", page=88)],
    )
    assert debug["raw_value"] == "42,1"
    assert debug["raw_unit"] == "milyar TL"
    assert debug["normalization_error"] is None


def test_build_answer_failed_normalisation_keeps_text_and_nulls_value():
    answer, debug = build_answer(
        gen(answer="Net kâr 3,1 milyar USD [1].", value="3,1", unit="milyar USD", citations=[1]),
        SOURCES,
    )
    assert answer.answer == "Net kâr 3,1 milyar USD [1]."
    assert (answer.value, answer.unit) == (None, None)
    assert "usd" in debug["normalization_error"].lower()
    assert debug["raw_value"] == "3,1"


# --- behaviour fields ---------------------------------------------------------


def test_build_answer_abstention_has_no_value():
    answer, debug = build_answer(
        gen(answer="Kaynaklarda Ziraat Bankası yok.", abstained=True, value="5", unit="TL"),
        SOURCES,
    )
    assert answer.abstained is True
    assert (answer.value, answer.unit, answer.clarification) == (None, None, None)
    assert debug["normalization_error"] is None


def test_build_answer_clarification():
    answer, _ = build_answer(
        gen(answer="Hangi banka?", clarification="  Hangi banka ve hangi yıl?  "), SOURCES
    )
    assert answer.clarification == "Hangi banka ve hangi yıl?"
    assert answer.abstained is False
    assert answer.value is None


def test_build_answer_blank_clarification_is_none():
    answer, _ = build_answer(gen(clarification="   "), SOURCES)
    assert answer.clarification is None


def test_build_answer_assumption_keeps_basis():
    answer, _ = build_answer(
        gen(
            answer="Soruda belirtilmediği için konsolide rakamı verdim: %18,2 [3].",
            value="18,2",
            unit="%",
            basis="konsolide",
            citations=[3],
        ),
        SOURCES,
    )
    assert answer.basis == "konsolide"
    assert (answer.value, answer.unit) == (18.2, "%")
    assert answer.citations == [Citation(doc="garanti-2025", page=50)]
