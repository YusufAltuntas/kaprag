"""Derived questions (eval/README.md §7 step 4): all arithmetic in Python, never the LLM.

- `compose_from_probes`: `karsilastirma` (ranking of 3+ banks, pairwise "which is
  higher") and `hesaplama` ("how much higher, in %/points") from checked probe facts
  of the SAME metric, year and basis. Cross-year comparisons are not composed from
  different reports: §4 says a change between years is read from the most recent
  report's comparative column, which the page-based YoY questions (`yoy_expected`) do.
- `yoy_expected`: the expected value of a year-over-year change from the two figures
  of one page (current and prior-year column).

Consolidated and solo figures are never mixed, and a money/ratio figure of unknown
basis is never used (`common_basis`). Near-ties (within 1%) are not ranked, so
rounding in a report cannot flip the gold answer.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterable, Sequence
from decimal import ROUND_CEILING, ROUND_HALF_UP, Decimal
from itertools import combinations

from kaprag.eval.schema import Evidence, Expected
from kaprag.gold.draft import (
    BANK_GENITIVE,
    BANK_NAMES,
    COMPUTED_PERCENT_TOLERANCE,
    TOLERANCE_BY_UNIT,
    Draft,
)
from kaprag.gold.metrics import BASIS_ADJ, METRICS_BY_ID, time_phrase

MIN_GAP_REL = Decimal("0.01")
_CENT = Decimal("0.01")


class BasisError(ValueError):
    """Sources of a derived question do not share one known basis."""


# --- arithmetic ------------------------------------------------------------------------


def _dec(value: object) -> Decimal:
    return value if isinstance(value, Decimal) else Decimal(str(value))


def percent_change(old: object, new: object) -> Decimal:
    """(new - old) / |old| * 100, rounded half-up to 2 decimals."""
    old_d, new_d = _dec(old), _dec(new)
    if old_d == 0:
        raise ValueError("percent change from zero is undefined")
    return ((new_d - old_d) / abs(old_d) * 100).quantize(_CENT, rounding=ROUND_HALF_UP)


def point_change(old: object, new: object) -> Decimal:
    """new - old for ratios already in %, rounded half-up to 2 decimals."""
    return (_dec(new) - _dec(old)).quantize(_CENT, rounding=ROUND_HALF_UP)


def common_basis(drafts: Sequence[Draft]) -> str | None:
    """The one basis all sources share; None only for metrics without a basis (counts).

    Raises BasisError when bases are mixed, or a money/ratio source has no basis.
    """
    bases = {d.basis for d in drafts}
    if len(bases) > 1:
        raise BasisError(f"mixed bases {sorted(map(str, bases))}")
    basis = bases.pop() if bases else None
    if basis is None and any(d.expected.unit in ("TL", "%") for d in drafts):
        raise BasisError("money/ratio figure of unknown basis")
    return basis


def abs_tolerance(d: Draft) -> Decimal:
    """A draft's tolerance as an absolute amount in its unit."""
    t = d.expected.tolerance
    if t is None:
        return Decimal(0)
    amount = _dec(t.amount)
    return amount * abs(_dec(d.expected.value)) if t.type == "rel" else amount


def rank(drafts: Sequence[Draft], min_gap_rel: Decimal = MIN_GAP_REL) -> list[Draft] | None:
    """Drafts by value, highest first; None if two neighbours are within `min_gap_rel` of
    each other or within their combined tolerances (rounded figures could swap)."""
    ordered = sorted(drafts, key=lambda d: (-_dec(d.expected.value), d.key))
    for hi, lo in zip(ordered, ordered[1:], strict=False):
        a, b = _dec(hi.expected.value), _dec(lo.expected.value)
        gap = max(min_gap_rel * max(abs(a), abs(b)), abs_tolerance(hi) + abs_tolerance(lo))
        if abs(a - b) <= gap:
            return None
    return ordered


def _rounding_excess(d: Draft) -> Decimal:
    """How far a source's tolerance was widened beyond the spec rule (printed rounding)."""
    t = d.expected.tolerance
    base = TOLERANCE_BY_UNIT[d.expected.unit]
    if t is None or d.expected.unit is None:
        return Decimal(0)
    return max(Decimal(0), _dec(t.amount) - _dec(base.amount))


# --- formatting ------------------------------------------------------------------------


def _tr_number(value: Decimal, decimals: int) -> str:
    text = f"{value:,.{decimals}f}"  # 1,234,567.89
    return text.replace(",", "_").replace(".", ",").replace("_", ".")


def format_value(value: object, unit: str | None) -> str:
    """Turkish rendering of a canonical value: '45,2 milyar TL (45.200.000.000 TL)'."""
    v = _dec(value)
    if unit == "%":
        return f"%{_tr_number(v, 2)}"
    if unit == "TL":
        exact = f"{_tr_number(v, 0)} TL"
        for scale, word in ((Decimal(10) ** 9, "milyar"), (Decimal(10) ** 6, "milyon")):
            if abs(v) >= scale:
                return f"{_tr_number(v / scale, 1)} {word} TL ({exact})"
        return exact
    return _tr_number(v, 0)


# --- year-over-year from one page -------------------------------------------------------


def yoy_expected(old: Decimal, new: Decimal, unit: str, old_year: int, new_year: int) -> Expected:
    """Expected change from `old_year` to `new_year`: % change, or points for ratios."""
    span = f"{old_year}: {format_value(old, unit)} → {new_year}: {format_value(new, unit)}"
    if unit == "%":
        delta = point_change(old, new)
        word = "arttı" if delta > 0 else "azaldı" if delta < 0 else "değişmedi"
        answer = f"{_tr_number(abs(delta), 2)} puan {word} ({span})."
        return Expected(
            answer=answer, value=float(delta), unit="%", tolerance=TOLERANCE_BY_UNIT["%"]
        )
    change = percent_change(old, new)
    word = "artış" if change > 0 else "azalış" if change < 0 else "değişim yok"
    answer = f"%{_tr_number(abs(change), 2)} {word} ({span})."
    return Expected(
        answer=answer, value=float(change), unit="%", tolerance=COMPUTED_PERCENT_TOLERANCE
    )


# --- composition from probes ------------------------------------------------------------


def _evidence(drafts: Iterable[Draft]) -> list[Evidence]:
    seen: dict[tuple[str, int | None], Evidence] = {}
    for d in drafts:
        for e in d.evidence:
            seen.setdefault((e.doc, e.page), e)
    return list(seen.values())


def _names(banks: Sequence[str]) -> str:
    names = [BANK_NAMES[b] for b in banks]
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " ve " + names[-1]


def _coin(key: str) -> bool:
    return hashlib.sha256(key.encode()).digest()[0] % 2 == 0


def _derived(
    key: str,
    type_: str,
    sources: Sequence[Draft],
    question: str,
    expected: Expected,
    basis: str | None,
    metric: str,
) -> Draft:
    banks = sorted({b for s in sources for b in s.banks})
    years = sorted({y for s in sources for y in s.years})
    return Draft(
        key=key,
        source="derived",
        type=type_,
        question=question,
        banks=banks,
        years=years,
        basis=basis,
        expected_behavior="answer",
        expected=expected,
        evidence=_evidence(sources),
        derived_from=sorted(s.key for s in sources),
        llm_checked=all(s.llm_checked for s in sources),
        notes="derived in code from probe facts; value computed in Python",
        metric=metric,
    )


def _ranking(metric_id: str, year: int, basis: str | None, group: list[Draft]) -> Draft | None:
    ordered = rank(group)
    if ordered is None:
        return None
    metric = METRICS_BY_ID[metric_id]
    banks = sorted(d.banks[0] for d in group)
    key = f"derived:rank:{metric_id}:{year}:{basis}:{'-'.join(banks)}"
    highest = _coin(key)
    if not highest:
        ordered = list(reversed(ordered))
    word = "yüksek" if highest else "düşük"
    question = (
        f"{time_phrase(metric, year)}, {_names(banks)} arasında {BASIS_ADJ[basis]}{metric.noun} "
        f"en {word} olan banka hangisiydi?"
    )
    listing = "; ".join(
        f"{BANK_NAMES[d.banks[0]]}: {format_value(d.expected.value, metric.unit)}" for d in ordered
    )
    expected = Expected(
        answer=f"{BANK_NAMES[ordered[0].banks[0]]}. Sıralama: {listing}.",
        value=None,
        unit=None,
        tolerance=None,
    )
    return _derived(key, "karsilastirma", group, question, expected, basis, metric_id)


def _pair(metric_id: str, year: int, basis: str | None, a: Draft, b: Draft) -> list[Draft]:
    """A 'which is higher' comparison and a 'how much higher' calculation for two banks."""
    metric = METRICS_BY_ID[metric_id]
    ordered = rank([a, b])
    hi, lo = sorted([a, b], key=lambda d: (-_dec(d.expected.value), d.key))
    bank_hi, bank_lo = hi.banks[0], lo.banks[0]
    pair_banks = sorted([bank_hi, bank_lo])
    tag = f"{metric_id}:{year}:{basis}:{'-'.join(pair_banks)}"
    v_hi, v_lo = _dec(hi.expected.value), _dec(lo.expected.value)
    out: list[Draft] = []
    if ordered is not None:
        question = (
            f"{time_phrase(metric, year)}, {_names(pair_banks)} karşılaştırıldığında "
            f"hangisinin {BASIS_ADJ[basis]}{metric.noun} daha yüksekti?"
        )
        expected = Expected(
            answer=f"{BANK_NAMES[bank_hi]} ({format_value(v_hi, metric.unit)}; "
            f"{BANK_NAMES[bank_lo]}: {format_value(v_lo, metric.unit)}).",
            value=None,
            unit=None,
            tolerance=None,
        )
        out.append(
            _derived(
                f"derived:pair:{tag}",
                "karsilastirma",
                [hi, lo],
                question,
                expected,
                basis,
                metric_id,
            )
        )
    if v_lo > 0 and v_hi != v_lo:
        lead = f"{time_phrase(metric, year)} {BANK_GENITIVE[bank_hi]} {BASIS_ADJ[basis]}"
        if metric.unit == "%":
            delta = point_change(v_lo, v_hi)
            question = f"{lead}{metric.noun}, {BANK_GENITIVE[bank_lo]}kinden kaç puan yüksekti?"
            answer = f"{_tr_number(delta, 2)} puan"
            extra = _rounding_excess(hi) + _rounding_excess(lo)
            tolerance = TOLERANCE_BY_UNIT["%"].model_copy(
                update={"amount": float(_dec(TOLERANCE_BY_UNIT["%"].amount) + extra)}
            )
        else:
            delta = percent_change(v_lo, v_hi)
            question = f"{lead}{metric.noun}, {BANK_GENITIVE[bank_lo]}kinden yüzde kaç fazlaydı?"
            answer = f"%{_tr_number(delta, 2)}"
            # relative rounding of the inputs propagates into the % difference
            extra = (v_hi / v_lo) * (_rounding_excess(hi) + _rounding_excess(lo)) * 100
            amount = (_dec(COMPUTED_PERCENT_TOLERANCE.amount) + extra).quantize(
                _CENT, rounding=ROUND_CEILING
            )
            tolerance = COMPUTED_PERCENT_TOLERANCE.model_copy(update={"amount": float(amount)})
        answer += (
            f" ({BANK_NAMES[bank_hi]}: {format_value(v_hi, metric.unit)}; "
            f"{BANK_NAMES[bank_lo]}: {format_value(v_lo, metric.unit)})."
        )
        expected = Expected(answer=answer, value=float(delta), unit="%", tolerance=tolerance)
        out.append(
            _derived(
                f"derived:gap:{tag}", "hesaplama", [hi, lo], question, expected, basis, metric_id
            )
        )
    return out


def compose_from_probes(probes: Iterable[Draft]) -> list[Draft]:
    """All comparison/calculation candidates from checked probe facts, sorted by key.

    Groups facts by (metric, year, basis) - one figure per bank - and composes a
    ranking for 3+ banks plus a pairwise comparison and gap for each pair. Facts with
    an unknown basis on a money/ratio metric are skipped.
    """
    groups: dict[tuple[str, int, str | None], dict[str, Draft]] = defaultdict(dict)
    for d in sorted(probes, key=lambda d: d.key):
        if d.metric is None or d.expected.value is None or not d.llm_checked:
            continue
        if METRICS_BY_ID[d.metric].needs_basis and d.basis is None:
            continue
        groups[(d.metric, d.years[0], d.basis)].setdefault(d.banks[0], d)
    out: list[Draft] = []
    for (metric_id, year, basis), by_bank in sorted(groups.items(), key=lambda kv: str(kv[0])):
        group = [by_bank[b] for b in sorted(by_bank)]
        if len(group) < 2:
            continue
        if common_basis(group) != basis:  # guard: groups are built per basis
            raise AssertionError("group basis mismatch")
        if len(group) >= 3 and (r := _ranking(metric_id, year, basis, group)) is not None:
            out.append(r)
        for a, b in combinations(group, 2):
            out += _pair(metric_id, year, basis, a, b)
    return sorted(out, key=lambda d: d.key)
