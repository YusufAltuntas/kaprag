"""The router's `compute` node: all arithmetic on the look-ups' figures, in Python.

Inputs are `Operand`s: one look-up's figure already normalised to canonical units by
the simple path's post-processing (`kaprag.rag.postprocess`: integer TL, %, adet, or
None for a bare number). `compute(kind, operands)` checks, in order:

1. Missing: an operand with no value (the look-up abstained, asked for clarification,
   failed, or its figure could not be normalised) -> not ok, `missing` names it. No
   guessing, no partial result.
2. Units: all operands must share one unit; a bare number counts as `adet` (as in the
   scorer, `kaprag.eval.numeric.compare`). TL is never combined with %.
3. Basis: for money (TL) and ratios (%), all operands' bases must be the same known
   basis or all unknown; konsolide next to solo, or a known basis next to an unknown
   one, is refused. Counts ignore the basis.
4. The operation (see `kaprag.agent.plan`), on `Decimal`s:
   `percent_change` [base, target] -> (target - base) / |base| x 100, %, 2 places;
   `difference` [a, b] -> a - b in the inputs' unit (TL and adet as integers, % as
   percentage points with 2 places); `ratio` [a, b] -> a / b, unit-less, 4 places;
   `share` [a, b] -> a / b x 100, %, 2 places; `max` / `min` -> every operand with
   the extreme value (ties: all of them, in plan order) and that value; `rank` ->
   operand ids ordered by value (ties keep plan order), value = the first one's.
   Division by zero is refused. Rounding is half-up.

A refusal (`refusal`, Turkish, shown to the user) or a missing operand makes the
question's answer an abstention; the graph never computes in the model.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

from kaprag.eval.numeric import to_json_number

PERCENT_PLACES = 2
RATIO_PLACES = 4
BASIS_UNITS = frozenset({"TL", "%"})  # units whose figures differ between konsolide and solo


@dataclass(frozen=True)
class Operand:
    """One look-up's figure. `label` names it for the user (bank, year, metric);
    `missing` says why there is no value, when there is none."""

    id: str
    label: str
    value: int | float | None
    unit: str | None
    basis: str | None
    missing: str | None = None


@dataclass(frozen=True)
class Computation:
    """The result of `compute`. `ok` is False when an operand is missing (`missing`
    holds their labels) or the computation was refused (`refusal`)."""

    kind: str
    ok: bool
    value: int | float | None = None
    unit: str | None = None
    basis: str | None = None
    winners: tuple[str, ...] = ()
    order: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()
    refusal: str | None = None

    def to_debug(self) -> dict[str, Any]:
        data = asdict(self)
        return {k: list(v) if isinstance(v, tuple) else v for k, v in data.items()}


def compute(kind: str, operands: Sequence[Operand], *, descending: bool = True) -> Computation:
    """Apply `kind` to the operands (in operation order); see the module docstring."""
    if missing := tuple(o.label for o in operands if o.value is None or o.missing):
        return Computation(kind, ok=False, missing=missing)
    units = {_unit(o) for o in operands}
    if len(units) > 1:
        shown = ", ".join(sorted(u or "birimsiz" for u in units))
        return Computation(kind, ok=False, refusal=f"rakamların birimleri uyuşmuyor ({shown})")
    unit = units.pop()
    basis, basis_error = _common_basis(operands, unit)
    if basis_error is not None:
        return Computation(kind, ok=False, refusal=basis_error)

    values = [Decimal(str(o.value)) for o in operands]
    if kind in ("max", "min", "rank"):
        return _select(kind, operands, values, unit, basis, descending)
    a, b = values
    if kind == "percent_change":
        if a == 0:
            return Computation(kind, ok=False, refusal="taban değer sıfır; yüzde değişim tanımsız")
        return _result(kind, (b - a) / abs(a) * 100, "%", basis, PERCENT_PLACES)
    if kind == "difference":
        places = 0 if unit in ("TL", "adet") else PERCENT_PLACES
        return _result(kind, a - b, unit, basis, places)
    if kind in ("ratio", "share"):
        if b == 0:
            return Computation(kind, ok=False, refusal="bölen değer sıfır; oran tanımsız")
        if kind == "ratio":
            return _result(kind, a / b, None, basis, RATIO_PLACES)
        return _result(kind, a / b * 100, "%", basis, PERCENT_PLACES)
    raise ValueError(f"unknown operation {kind!r}")


def _unit(operand: Operand) -> str | None:
    return "adet" if operand.unit is None else operand.unit


def _common_basis(operands: Sequence[Operand], unit: str | None) -> tuple[str | None, str | None]:
    """(common basis, refusal). Counts ignore the basis."""
    bases = {o.basis for o in operands}
    if unit not in BASIS_UNITS:
        return (bases.pop() if len(bases) == 1 else None), None
    known = bases - {None}
    if len(known) > 1:
        return None, "rakamların esasları uyuşmuyor (konsolide ve solo)"
    if known and None in bases:
        return None, "bazı rakamların esası (konsolide ya da solo) belirlenemedi"
    return (known.pop() if known else None), None


def _result(kind: str, value: Decimal, unit: str | None, basis: str | None, places: int):
    rounded = value.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)
    number = int(rounded) if places == 0 else to_json_number(rounded)
    return Computation(kind, ok=True, value=number, unit=unit, basis=basis)


def _select(
    kind: str,
    operands: Sequence[Operand],
    values: Sequence[Decimal],
    unit: str | None,
    basis: str | None,
    descending: bool,
) -> Computation:
    if kind == "rank":
        order = sorted(range(len(operands)), key=lambda i: -values[i] if descending else values[i])
        first = order[0]
        return Computation(
            kind,
            ok=True,
            value=operands[first].value,
            unit=unit,
            basis=basis,
            order=tuple(operands[i].id for i in order),
        )
    best = max(values) if kind == "max" else min(values)
    winners = tuple(o.id for o, v in zip(operands, values, strict=True) if v == best)
    winner = next(o for o in operands if o.id == winners[0])
    return Computation(kind, ok=True, value=winner.value, unit=unit, basis=basis, winners=winners)


# --- Turkish formatting -----------------------------------------------------------


def format_number(value: int | float) -> str:
    """Turkish format: `.` groups thousands, `,` is the decimal mark (`1.234,5`)."""
    number = Decimal(str(value))
    sign = "-" if number < 0 else ""
    text = format(abs(number), "f")
    whole, _, fraction = text.partition(".")
    grouped = f"{int(whole):,}".replace(",", ".")
    return f"{sign}{grouped},{fraction}" if fraction else f"{sign}{grouped}"


def format_quantity(value: int | float, unit: str | None) -> str:
    """A canonical value for the user: `%48,99`, `-%26,59`, `18.897.000.000 TL`, `1.034`."""
    if unit == "%":
        text = format_number(abs(value))
        return f"-%{text}" if value < 0 else f"%{text}"
    text = format_number(value)
    return f"{text} TL" if unit == "TL" else text
