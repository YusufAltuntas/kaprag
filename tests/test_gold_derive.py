"""Derived questions: Python arithmetic, basis rules and composition from probe facts."""

from decimal import Decimal

import pytest

from kaprag.eval.schema import Evidence, Expected
from kaprag.gold.derive import (
    BasisError,
    common_basis,
    compose_from_probes,
    percent_change,
    point_change,
    rank,
    yoy_expected,
)
from kaprag.gold.draft import TOLERANCE_BY_UNIT, Draft


def probe(bank, year, value, *, metric="net_profit", unit="TL", basis="konsolide", page=10):
    return Draft(
        key=f"probe:{bank}-{year}:{metric}",
        source="probe",
        type="tablo" if unit != "adet" else "olgusal",
        question=f"{bank} {year} {metric}?",
        banks=[bank],
        years=[year],
        basis=basis,
        expected_behavior="answer",
        expected=Expected(
            answer=str(value), value=value, unit=unit, tolerance=TOLERANCE_BY_UNIT[unit]
        ),
        evidence=[Evidence(doc=f"{bank}-{year}", page=page)],
        metric=metric,
        llm_checked=True,
    )


def test_percent_change_rounds_to_two_decimals():
    assert percent_change(100, Decimal("112.5")) == Decimal("12.50")
    assert percent_change(200, 150) == Decimal("-25.00")
    assert percent_change(3, 4) == Decimal("33.33")
    assert percent_change(3, 5) == Decimal("66.67")  # half up


def test_percent_change_uses_absolute_base_and_refuses_zero():
    assert percent_change(-50, 50) == Decimal("200.00")
    with pytest.raises(ValueError):
        percent_change(0, 10)


def test_point_change_for_ratios():
    assert point_change(Decimal("18.2"), Decimal("17.45")) == Decimal("-0.75")


def test_common_basis_refuses_mixing_and_unknown():
    a, b = probe("akbank", 2025, 10), probe("isbank", 2025, 20, basis="solo")
    with pytest.raises(BasisError):
        common_basis([a, b])
    with pytest.raises(BasisError):
        common_basis([a, probe("garanti", 2025, 5, basis=None)])
    assert common_basis([a, probe("isbank", 2025, 20)]) == "konsolide"


def test_common_basis_allows_unknown_for_counts():
    counts = [
        probe(b, 2025, 700, metric="branches", unit="adet", basis=None)
        for b in ("akbank", "isbank")
    ]
    assert common_basis(counts) is None


def test_rank_orders_high_to_low_and_refuses_near_ties():
    a, b, c = probe("akbank", 2025, 300), probe("isbank", 2025, 100), probe("garanti", 2025, 200)
    assert [d.banks[0] for d in rank([a, b, c])] == ["akbank", "garanti", "isbank"]
    close = probe("yapikredi", 2025, 299)  # within 1% of akbank
    assert rank([a, b, close]) is None


def test_yoy_expected_is_computed_in_python():
    exp = yoy_expected(Decimal("80000000000"), Decimal("100000000000"), "TL", 2024, 2025)
    assert exp.value == 25.0 and exp.unit == "%"
    assert exp.tolerance.type == "abs" and exp.tolerance.amount == 0.5
    assert "%25" in exp.answer
    ratio = yoy_expected(Decimal("18.2"), Decimal("17.45"), "%", 2024, 2025)
    assert ratio.value == -0.75 and ratio.unit == "%"
    assert "puan" in ratio.answer


def test_compose_ranking_and_gap_from_probes():
    probes = [
        probe("akbank", 2025, 300_000, page=5),
        probe("isbank", 2025, 100_000, page=7),
        probe("garanti", 2025, 200_000, page=9),
    ]
    drafts = compose_from_probes(probes)
    ranking = [d for d in drafts if d.type == "karsilastirma" and len(d.banks) == 3]
    assert ranking, drafts
    r = ranking[0]
    assert r.banks == ["akbank", "garanti", "isbank"]
    assert r.years == [2025] and r.basis == "konsolide"
    assert r.expected.value is None and r.expected_behavior == "answer"
    assert sorted(r.derived_from) == sorted(p.key for p in probes)
    assert {(e.doc, e.page) for e in r.evidence} == {
        ("akbank-2025", 5),
        ("isbank-2025", 7),
        ("garanti-2025", 9),
    }
    assert "konsolide" in r.question and "2025" in r.question
    top = "Akbank" if "en yüksek" in r.question else "İş Bankası"
    assert r.expected.answer.startswith(top)
    gaps = [d for d in drafts if d.type == "hesaplama"]
    akbank_vs_isbank = [d for d in gaps if set(d.banks) == {"akbank", "isbank"}]
    assert akbank_vs_isbank[0].expected.value == 200.0  # 300k is 200% above 100k
    assert akbank_vs_isbank[0].expected.unit == "%"


def test_compose_never_mixes_bases_or_unknown_basis():
    probes = [
        probe("akbank", 2025, 300),
        probe("isbank", 2025, 100, basis="solo"),
        probe("garanti", 2025, 200, basis=None),
    ]
    assert compose_from_probes(probes) == []
    probes.append(probe("yapikredi", 2025, 150, basis="solo"))
    drafts = compose_from_probes(probes)
    assert drafts
    for d in drafts:
        assert set(d.banks) == {"isbank", "yapikredi"} and d.basis == "solo"


def test_compose_does_not_compare_years_across_reports():
    """Changes between years come from one report's comparative column (spec §4), so
    probe facts from different reports are never combined over years."""
    probes = [
        probe("akbank", y, v, metric="branches", unit="adet", basis=None)
        for y, v in ((2023, 715), (2024, 700), (2025, 690))
    ]
    assert compose_from_probes(probes) == []


def test_compose_is_deterministic():
    probes = [probe("akbank", 2025, 300), probe("isbank", 2025, 100), probe("garanti", 2025, 200)]
    assert compose_from_probes(probes) == compose_from_probes(list(reversed(probes)))


def test_rank_refuses_values_within_their_combined_tolerances():
    a, b = probe("akbank", 2025, 1000), probe("isbank", 2025, 985)  # 1.5% apart, 1% each
    assert rank([a, b]) is None
    assert rank([a, probe("isbank", 2025, 970)]) is not None  # 3% apart


def test_gap_tolerance_grows_with_rounded_inputs():
    from kaprag.eval.schema import Tolerance

    hi = probe("akbank", 2025, 2_600_000)
    hi = hi.model_copy(
        update={
            "expected": hi.expected.model_copy(
                update={"tolerance": Tolerance(type="rel", amount=0.0193)}
            )
        }
    )
    lo = probe("isbank", 2025, 1_000_000)
    gap = next(d for d in compose_from_probes([hi, lo]) if d.type == "hesaplama")
    assert gap.expected.value == 160.0
    assert gap.expected.tolerance.amount == 2.92  # 0.5 + 2.6 * 0.0093 * 100, rounded up
    exact = compose_from_probes([probe("akbank", 2025, 2_600_000), lo])
    assert next(d for d in exact if d.type == "hesaplama").expected.tolerance.amount == 0.5


def test_gap_tolerance_scales_an_absolute_count_excess_by_the_source_value():
    """21 000 +- 500 employees vs 15 000 (exact): the true gap lies in
    [20 500 / 15 000 - 1, 21 500 / 15 000 - 1] = [36.67%, 43.33%], i.e. +-3.33 points,
    so the tolerance is 0.5 + 1.4 * (500 / 21 000) * 100 = 3.8333 -> 3.84."""
    from kaprag.eval.schema import Tolerance

    hi = probe("akbank", 2023, 21_000, metric="employees", unit="adet", basis=None)
    hi = hi.model_copy(
        update={
            "expected": hi.expected.model_copy(
                update={"tolerance": Tolerance(type="abs", amount=500)}
            )
        }
    )
    lo = probe("yapikredi", 2023, 15_000, metric="employees", unit="adet", basis=None)
    gap = next(d for d in compose_from_probes([hi, lo]) if d.type == "hesaplama")
    assert gap.expected.value == 40.0
    assert gap.expected.tolerance.type == "abs" and gap.expected.tolerance.amount == 3.84


def test_rounding_excess_is_zero_without_a_unit():
    from kaprag.gold.derive import _rounding_excess

    d = probe("akbank", 2025, 1)
    d = d.model_copy(
        update={"expected": Expected(answer="x", value=None, unit=None, tolerance=None)}
    )
    assert _rounding_excess(d) == 0
