"""Probe validation, cevapsiz checks, belirsiz derivation, assembly and the stats report."""

from collections import Counter

import pytest
from gold_helpers import write_parsed

from kaprag.eval.schema import Evidence, Expected, dump_question, parse_question
from kaprag.gold.abstain import (
    AbstainOut,
    AbstainVerdict,
    doc_filter,
    keep_reason,
    to_draft,
    validate_candidate,
)
from kaprag.gold.ambiguous import AmbiguousTask, select_sources
from kaprag.gold.ambiguous import to_draft as ambiguous_draft
from kaprag.gold.assemble import assemble, balanced_pick, select
from kaprag.gold.draft import TOLERANCE_BY_UNIT, Draft
from kaprag.gold.figures import FigureError
from kaprag.gold.metrics import METRICS_BY_ID
from kaprag.gold.pages import DocPages
from kaprag.gold.probes import ProbeOut, collect_passages, validate_probe
from kaprag.gold.stats import distribution_report
from kaprag.retrieval import SearchHit


def hit(cid, doc="akbank-2024", start=3, end=3):
    return SearchHit(cid, doc, start, end, 1.0, "text", (), f"text {cid}")


def single(
    key,
    bank="akbank",
    year=2024,
    value=100,
    unit="TL",
    basis="konsolide",
    type_="tablo",
    checked=True,
):
    return Draft(
        key=key,
        source="probe",
        type=type_,
        question=f"{bank} {year} konsolide net kâr?" if basis else f"{bank} {year} şube?",
        banks=[bank],
        years=[year],
        basis=basis,
        expected_behavior="answer",
        expected=Expected(answer="x", value=value, unit=unit, tolerance=TOLERANCE_BY_UNIT[unit]),
        evidence=[Evidence(doc=f"{bank}-{year}", page=5)],
        llm_checked=checked,
        metric="net_profit",
    )


# --- probes -------------------------------------------------------------------------------


def test_collect_passages_dedupes_in_metric_order():
    hits = {"a": [hit("1"), hit("2"), hit("3"), hit("4")], "b": [hit("2"), hit("5")]}
    assert [h.chunk_id for h in collect_passages(hits, per_metric=3)] == ["1", "2", "3", "5"]


def probe_out(**kw):
    base = dict(
        metric="net_profit",
        found=True,
        passage=1,
        figure="45.210.300",
        unit_text="bin TL",
        value=45_210_300_000,
        unit="TL",
        basis="konsolide",
        answer="45,2 milyar TL",
        components=[],
    )
    return ProbeOut(**(base | kw))


def test_probe_resolves_page_and_words_question(tmp_dir):
    pages = DocPages.load("akbank-2024", write_parsed(tmp_dir, "akbank-2024"))
    metric = METRICS_BY_ID["net_profit"]
    passages = [hit("c1", start=2, end=3)]
    d = validate_probe(probe_out(), metric, "akbank-2024", "akbank", 2024, passages, pages)
    assert d.evidence[0].page == 3  # the page of the span that holds the figure
    assert d.key == "probe:akbank-2024:net_profit" and d.metric == "net_profit"
    assert d.question == "Akbank, 2024 yılında konsolide bazda ne kadar net kâr elde etti?"
    assert d.expected.value == 45_210_300_000 and not d.llm_checked
    with pytest.raises(FigureError):
        validate_probe(
            probe_out(figure="99.999.999", value=99_999_999_000),
            metric,
            "akbank-2024",
            "akbank",
            2024,
            passages,
            pages,
        )
    with pytest.raises(ValueError, match="not found"):
        validate_probe(
            probe_out(found=False), metric, "akbank-2024", "akbank", 2024, passages, pages
        )
    with pytest.raises(ValueError, match="passage"):
        validate_probe(probe_out(passage=2), metric, "akbank-2024", "akbank", 2024, passages, pages)


# --- cevapsiz -------------------------------------------------------------------------------


def abstain_out(**kw):
    base = dict(
        kind="korpus_disi_banka",
        question="Ziraat Bankası 2024'te kaç şubeye sahipti?",
        banks=[],
        years=[2024],
        why="korpusta Ziraat yok",
    )
    return AbstainOut(**(base | kw))


def test_abstain_candidate_validation():
    validate_candidate(abstain_out())
    validate_candidate(
        abstain_out(
            kind="yanlis_oncul", banks=["akbank"], question="Akbank 2024'te ne kadar zarar etti?"
        )
    )
    with pytest.raises(ValueError):
        validate_candidate(abstain_out(question="Ziraat ve Akbank 2024'te kaç şube açtı?"))
    with pytest.raises(ValueError):
        validate_candidate(abstain_out(years=[2023]))
    with pytest.raises(ValueError):
        validate_candidate(abstain_out(kind="olmayan_bilgi", banks=["garanti"]))


def test_abstain_doc_filter_and_keep_rules():
    assert doc_filter(abstain_out()) is None
    assert doc_filter(abstain_out(banks=["akbank"])) == [
        "akbank-2023",
        "akbank-2024",
        "akbank-2025",
    ]
    v = AbstainVerdict(index=0, verdict="not_in_corpus", evidence_passages=[], reason="")
    assert keep_reason(abstain_out(), v) is None
    assert "answerable" in keep_reason(
        abstain_out(), v.model_copy(update={"verdict": "answerable"})
    )
    premise = abstain_out(kind="yanlis_oncul", banks=["akbank"])
    assert keep_reason(premise, v) is not None  # a false premise must be contradicted
    fp = v.model_copy(update={"verdict": "false_premise", "evidence_passages": [1, 9]})
    assert keep_reason(premise, fp) is None
    assert keep_reason(abstain_out(), None) == "no judge verdict"
    d = to_draft("abstain:0", premise, fp, [hit("c", start=4, end=5)])
    assert d.expected_behavior == "abstain" and d.type == "cevapsiz"
    assert [(e.doc, e.page) for e in d.evidence] == [("akbank-2024", 4), ("akbank-2024", 5)]


# --- belirsiz -------------------------------------------------------------------------------


def test_ambiguous_sources_are_balanced_and_used_once():
    singles = [single(f"s{b}{i}", bank=b) for b in ("akbank", "isbank") for i in range(3)]
    singles.append(single("solo", basis="solo"))
    singles.append(single("unchecked", checked=False))
    tasks = select_sources(singles, per_kind=2)
    assert len(tasks) == 6
    assert len({t.source.key for t in tasks}) == 6
    assert all(t.source.basis == "konsolide" for t in tasks if t.kind == "drop_basis")
    assert "unchecked" not in {t.source.key for t in tasks}
    assert Counter(t.source.banks[0] for t in tasks if t.kind == "drop_bank") == {
        "akbank": 1,
        "isbank": 1,
    }
    assert tasks == select_sources(list(reversed(singles)), per_kind=2)


def test_ambiguous_derivation_rules():
    src = single("s").model_copy(
        update={"question": "Akbank 2024 yılında konsolide bazda ne kadar net kâr elde etti?"}
    )
    d = ambiguous_draft(
        AmbiguousTask("drop_bank", src),
        "2024 yılında konsolide bazda ne kadar net kâr elde edildi?",
    )
    assert d.expected_behavior == "clarify" and d.banks == [] and d.years == [2024]
    assert not d.llm_checked  # nothing a page check could confirm
    assert d.derived_from == ["s"] and d.expected.value is None
    d = ambiguous_draft(
        AmbiguousTask("drop_year", src), "Akbank konsolide bazda ne kadar net kâr elde etti?"
    )
    assert d.expected_behavior == "clarify" and d.banks == ["akbank"] and d.years == []
    d = ambiguous_draft(
        AmbiguousTask("drop_basis", src), "Akbank 2024 yılında ne kadar net kâr elde etti?"
    )
    assert d.expected_behavior == "answer_with_assumption" and d.basis == "konsolide"
    assert d.expected.value == src.expected.value and d.evidence == src.evidence
    assert d.llm_checked  # the answer is the checked source answer
    with pytest.raises(ValueError):
        ambiguous_draft(AmbiguousTask("drop_bank", src), src.question.replace("2024", "2024 "))
    with pytest.raises(ValueError):
        ambiguous_draft(AmbiguousTask("drop_year", src), "İş Bankası 2024'te ne kadar kâr etti?")
    with pytest.raises(ValueError):
        ambiguous_draft(AmbiguousTask("drop_basis", src), "Akbank 2024 konsolide kârı?")


# --- assembly -------------------------------------------------------------------------------


def derived(key, sources, type_="karsilastirma"):
    return Draft(
        key=key,
        source="derived",
        type=type_,
        question=key,
        banks=["akbank", "isbank"],
        years=[2024],
        basis="konsolide",
        expected_behavior="answer",
        expected=Expected(answer="a", value=None, unit=None, tolerance=None),
        evidence=[],
        derived_from=[s.key for s in sources],
        llm_checked=True,
        metric="net_profit",
    )


def test_balanced_pick_round_robins_over_cells():
    pool = [single(f"a{i}", bank="akbank") for i in range(5)] + [single("i0", bank="isbank")]
    picked = balanced_pick(pool, 2)
    assert {d.banks[0] for d in picked} == {"akbank", "isbank"}
    assert balanced_pick(pool, 2) == balanced_pick(list(reversed(pool)), 2)


def test_select_forces_sources_and_honours_quotas():
    a, b = single("a", bank="akbank"), single("b", bank="isbank")
    extra = [single(f"x{i}", bank="garanti") for i in range(5)]
    chosen = select([a, b, *extra, derived("d1", [a, b])], {"tablo": 3, "karsilastirma": 1})
    keys = {d.key for d in chosen}
    assert {"a", "b", "d1"} <= keys
    assert sum(d.type == "tablo" for d in chosen) == 3


def test_assemble_ids_split_pilot_and_schema():
    singles = [
        single(f"s{b}{y}", bank=b, year=y)
        for b in ("akbank", "isbank", "garanti")
        for y in (2023, 2024, 2025)
    ]
    pool = singles + [derived("d1", singles[:2]), derived("d2", singles[3:5], "hesaplama")]
    pool[-1] = pool[-1].model_copy(
        update={
            "expected": Expected(answer="%5", value=5.0, unit="%", tolerance=TOLERANCE_BY_UNIT["%"])
        }
    )
    questions = assemble(pool, {"tablo": 9, "karsilastirma": 1, "hesaplama": 1}, pilot_size=4)
    assert [q.id for q in questions] == [f"q{i:03d}" for i in range(1, 12)]
    assert [q.type for q in questions][:9] == ["tablo"] * 9  # type order, then banks/years
    ids = {q.id for q in questions}
    for q in questions:
        assert q.author == "llm" and q.verified is False and q.verified_at is None
        assert set(q.derived_from) <= ids
        parse_question(dump_question(q))  # round-trips through the strict schema
    pilot = [q for q in questions if q.pilot]
    assert len(pilot) == min(4, sum(q.split == "dev" for q in questions))
    assert all(q.split == "dev" for q in pilot)
    assert questions == assemble(
        list(reversed(pool)), {"tablo": 9, "karsilastirma": 1, "hesaplama": 1}, pilot_size=4
    )
    report = distribution_report(questions)
    assert "verified: true = 0" in report and "type x split x bank" in report


def test_ambiguous_prefers_probe_sources():
    page = single("page1").model_copy(update={"source": "page"})
    probe = single("probe1")
    tasks = select_sources([page, probe], per_kind=1)
    assert tasks[0].source.key == "probe1"


# --- fix round 1 ---------------------------------------------------------------------------


def test_probe_with_unknown_basis_on_money_metric_is_dropped(tmp_dir):
    pages = DocPages.load("akbank-2024", write_parsed(tmp_dir, "akbank-2024"))
    with pytest.raises(ValueError, match="unknown basis"):
        validate_probe(
            probe_out(basis=None),
            METRICS_BY_ID["net_profit"],
            "akbank-2024",
            "akbank",
            2024,
            [hit("c1")],
            pages,
        )


def test_abstain_answer_does_not_repeat_the_label():
    premise = abstain_out(
        kind="yanlis_oncul", banks=["akbank"], why="Yanlış öncül: Akbank 2024'te kâr etti."
    )
    v = AbstainVerdict(index=0, verdict="false_premise", evidence_passages=[], reason="")
    assert to_draft("k", premise, v, []).expected.answer == (
        "Yanlış öncül: Akbank 2024'te kâr etti."
    )


def test_page_single_duplicating_a_probe_is_removed():
    from kaprag.gold.assemble import drop_probe_duplicates

    probe = single("probe:akbank-2024:net_profit")
    same = single("page:akbank-2024:5:0").model_copy(update={"source": "page"})
    other_page = same.model_copy(
        update={"key": "page:x", "evidence": [Evidence(doc="akbank-2024", page=6)]}
    )
    kept, dropped = drop_probe_duplicates([same, other_page], [probe])
    assert [d.key for d in kept] == ["page:x"]
    assert [(d.key, k) for d, k in dropped] == [("page:akbank-2024:5:0", probe.key)]
