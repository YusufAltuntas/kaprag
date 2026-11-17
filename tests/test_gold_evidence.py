"""Evidence completion: candidate pages, the check, and propagation to derived drafts."""

from gold_helpers import element

from kaprag.eval.schema import Evidence, Expected
from kaprag.gold.draft import TOLERANCE_BY_UNIT, Draft
from kaprag.gold.evidence import (
    MAX_CANDIDATES,
    Candidate,
    EvidenceCheck,
    PageVerdict,
    ScanIndex,
    apply_verdicts,
    build_prompt,
    candidates_for,
    figure_candidates,
    needs_completion,
    propagate,
)
from kaprag.gold.pages import DocPages
from kaprag.parsing.outputs import Element
from kaprag.retrieval import SearchHit

DOC = "akbank-2024"


def doc(texts: dict[int, str]) -> DocPages:
    elements = {p: [Element(**element(p, "paragraph", t))] for p, t in texts.items()}
    return DocPages(DOC, elements, {p: t for p, t in texts.items()})


def single(key="s", value=111_262_000_000, unit="TL", page=34, type_="tablo", **kw) -> Draft:
    tol = TOLERANCE_BY_UNIT[unit] if value is not None else None
    data = dict(
        key=key,
        source="probe",
        type=type_,
        question="Akbank 2024 net kâr?",
        banks=["akbank"],
        years=[2024],
        basis="konsolide",
        expected_behavior="answer",
        expected=Expected(
            answer="x", value=value, unit=unit if value is not None else None, tolerance=tol
        ),
        evidence=[Evidence(doc=DOC, page=page)],
    )
    return Draft(**(data | kw))


PAGES = {
    34: "Net kâr 111.262.000 bin TL",  # original
    5: "Konsolide net kârımız 111 milyar TL oldu.",  # rounded, within 1%
    6: "Net kâr 111.262.000 bin TL olarak gerçekleşti.",  # exact
    7: "Hedef 113 milyar TL.",  # 1.6% off: outside tolerance
    8: "111.262 ve 111 rakamları",  # no scale word on the page
    9: "Sayfa 111.262.000 TL",  # unscaled 111 million TL: wrong magnitude
}


def test_figure_scan_exact_and_rounded_within_tolerance():
    cands = figure_candidates(single(), ScanIndex(doc(PAGES)), exclude={34})
    assert [c.page for c in cands] == [6, 5]  # exact match ranks first
    assert all(c.source == "figure" for c in cands)
    assert "111 milyar" in cands[1].excerpt


def test_figure_scan_cap_and_small_counts():
    many = {p: "Net kâr 111.262.000 bin TL" for p in range(1, 20)}
    assert len(figure_candidates(single(page=1), ScanIndex(doc(many)), exclude={1})) == 4
    counts = {1: "7 şube", 2: "7 şube", 3: "694 şube", 4: "694 şube"}
    index = ScanIndex(doc(counts))
    assert figure_candidates(single(value=7, unit="adet", page=1), index, {1}) == []
    assert [
        c.page for c in figure_candidates(single(value=694, unit="adet", page=3), index, {3})
    ] == [4]


def test_yoy_calculation_needs_both_figures():
    yoy = single(
        type_="hesaplama",
        value=18.66,
        unit="%",
        quote="2024: 45.210.300; 2023: 38.100.000 (bin TL)",
    )
    texts = {3: "x", 4: "45.210.300 ve 38.100.000", 5: "yalnız 45.210.300"}
    assert [c.page for c in figure_candidates(yoy, ScanIndex(doc(texts)), {3})] == [4]


def hit(doc_id, page, text="passage"):
    return SearchHit(f"{doc_id}:{page}", doc_id, page, page, 1.0, "text", (), text)


def test_candidates_same_document_figures_first_and_capped():
    hits = [hit("isbank-2024", 2), hit(DOC, 34), hit(DOC, 6), *[hit(DOC, p) for p in range(40, 50)]]
    cands = candidates_for(single(), ScanIndex(doc(PAGES)), hits)
    assert [c.page for c in cands] == [6, 5, 40, 41, 42, 43]
    assert len(cands) == MAX_CANDIDATES
    assert [c.source for c in cands] == ["figure", "figure"] + ["search"] * 4
    text = single(value=None, type_="sentez", key_points=["a", "b"])
    assert [c.page for c in candidates_for(text, ScanIndex(doc(PAGES)), hits)][:2] == [6, 40]


def test_only_checked_pages_are_added_after_the_original():
    cands = [Candidate(6, "figure", "a"), Candidate(5, "figure", "b"), Candidate(40, "search", "c")]
    check = EvidenceCheck(
        items=[
            PageVerdict(candidate=0, states_answer=True, reason="aynı rakam"),
            PageVerdict(candidate=2, states_answer=False, reason="başka kalem"),
        ]
    )
    d, record = apply_verdicts(single(), cands, check)
    assert [e.page for e in d.evidence] == [34, 6]
    assert [a["page"] for a in record["added"]] == [6]
    assert {r["page"]: r["reason"] for r in record["rejected"]} == {
        5: "no verdict",
        40: "başka kalem",
    }


def test_failed_check_adds_nothing():
    d, record = apply_verdicts(single(), [Candidate(6, "figure", "a")], None)
    assert [e.page for e in d.evidence] == [34]
    assert record["rejected"][0]["reason"] == "check call failed"


def test_derived_and_assumption_drafts_take_the_union():
    a = single("a").model_copy(
        update={"evidence": [Evidence(doc=DOC, page=34), Evidence(doc=DOC, page=6)]}
    )
    b = single("b", page=12).model_copy(
        update={
            "banks": ["isbank"],
            "evidence": [Evidence(doc="isbank-2024", page=12), Evidence(doc="isbank-2024", page=3)],
        }
    )
    derived = single(
        "d",
        value=None,
        type_="karsilastirma",
        derived_from=["a", "b"],
        evidence=[Evidence(doc=DOC, page=34), Evidence(doc="isbank-2024", page=12)],
    )
    assume = single(
        "w", type_="belirsiz", derived_from=["a"], expected_behavior="answer_with_assumption"
    )
    clarify = single(
        "c",
        value=None,
        type_="belirsiz",
        derived_from=["a"],
        evidence=[],
        expected_behavior="clarify",
        basis=None,
    )
    out = {d.key: d for d in propagate([a, b, derived, assume, clarify])}
    assert [(e.doc, e.page) for e in out["d"].evidence] == [
        (DOC, 34),
        ("isbank-2024", 12),
        (DOC, 6),
        ("isbank-2024", 3),
    ]
    assert [e.page for e in out["w"].evidence] == [34, 6]
    assert out["c"].evidence == []
    assert needs_completion(a) and not needs_completion(derived) and not needs_completion(assume)


def test_prompt_lists_candidates_and_the_expected_answer():
    prompt = build_prompt(single(), [Candidate(6, "figure", "Net kâr 111.262.000 bin TL")])
    assert "### Aday 0 (sayfa 6)" in prompt and "111262000000 TL" in prompt
    assert "konsolide" in prompt
