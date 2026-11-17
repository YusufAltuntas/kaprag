"""Code validation of a figure an LLM read off a page.

The model returns the figure as written (`"1.073.924"`, `"(3.572.806)"`, `"%12,5"`,
`"4 trilyon 548 milyar"`), the unit text that applies to it on the page (`"bin TL"`,
`"milyon TL"`, `"%"`, `"şube"`) and its own normalised value. Here every number written
in the figure must occur on the page, the canonical value is recomputed with
`kaprag.eval.numeric`, and the model's value must agree with it - a disagreement means
the model and the code read the unit differently. A total the page only gives as parts
(`"693 yurt içi + 1 yurt dışı"` branches) is passed as `components` and summed here.
"""

from __future__ import annotations

import re
from collections.abc import Sequence, Set
from decimal import ROUND_CEILING, Decimal

from kaprag.eval.numeric import NumericParseError, normalize, parse_number, to_json_number
from kaprag.eval.schema import Tolerance
from kaprag.gold.draft import TOLERANCE_BY_UNIT, fold
from kaprag.gold.pages import NUMBER_TOKEN

VALUE_REL_EPS = Decimal("0.000001")
# Scale words of `kaprag.eval.numeric`; one of them in the unit text must be on the page.
SCALE_WORDS = ("bin", "milyon", "mn", "mio", "milyar", "mlr", "trilyon", "trn")
_SCALES = {"trilyon": 10**12, "milyar": 10**9, "milyon": 10**6, "bin": 10**3}
_COMPOUND_PART = re.compile(r"(\d[\d.,]*)\s*(trilyon|milyar|milyon|bin)\b")


class FigureError(ValueError):
    """The figure cannot be confirmed; the message is the drop reason."""


class UnitError(FigureError):
    """The figure is on the page but its unit has no canonical form (e.g. 'MW', 'genç')."""


def _parse(token: str, figure: str) -> Decimal:
    try:
        return parse_number(token)
    except NumericParseError as exc:
        raise FigureError(f"figure {figure!r}: {exc}") from None


def figure_amount(figure: str) -> tuple[Decimal, list[Decimal]]:
    """(amount, numbers written) of a figure: one number ('%12,5' -> 12.5, '(3.572)' ->
    -3572), or a compound amount in words ('4 trilyon 548 milyar' -> 4548 * 10**9)."""
    tokens = NUMBER_TOKEN.findall(figure)
    if len(tokens) == 1:
        number = _parse(tokens[0], figure)
        return number, [number]
    parts = _COMPOUND_PART.findall(fold(figure))
    if len(tokens) > 1 and len(parts) == len(tokens):
        scales = [_SCALES[s] for _, s in parts]
        if scales == sorted(scales, reverse=True) and len(set(scales)) == len(scales):
            numbers = [_parse(n, figure) for n, _ in parts]
            amount = sum((n * s for n, s in zip(numbers, scales, strict=True)), Decimal(0))
            return amount, numbers
    raise FigureError(f"figure {figure!r} does not hold exactly one number")


def figure_number(figure: str) -> Decimal:
    """The first number written in a figure (for locating its page)."""
    return figure_amount(figure)[1][0]


def read_figure(
    figure: str | None,
    unit_text: str | None,
    model_value: float | int | None,
    model_unit: str | None,
    page_numbers: Set[Decimal],
    page_text: str | None = None,
    components: Sequence[str] = (),
) -> tuple[int | float, str]:
    """Canonical (value, unit) of a figure (or of the sum of `components` when no single
    figure is given), checked against the page and the model.

    Raises FigureError when the figure is missing or a written number is not on the
    page, its scale word (bin/milyon/milyar...) is not written on the page (`page_text`,
    when given) - i.e. the multiplier was guessed - or the model's normalised value
    disagrees; UnitError when the unit text is not understood or not canonical.
    """
    parts = [figure] if figure else list(components)
    if not parts:
        raise FigureError("no figure given")
    amount = Decimal(0)
    for part in parts:
        value, numbers = figure_amount(part)
        for number in numbers:
            if number not in page_numbers and abs(number) not in page_numbers:
                raise FigureError(f"figure {part!r} not found on the page")
        amount += value
    shown = figure or " + ".join(parts)
    unit_words = unit_text or ""
    if "%" in shown and "%" not in unit_words:
        unit_words += " %"
    if page_text is not None:
        _check_scale_on_page(unit_words + " " + shown, page_text)
    try:
        quantity = normalize(amount, unit_words)
    except NumericParseError as exc:
        raise UnitError(f"unit {unit_text!r} not understood: {exc}") from None
    unit = quantity.unit or ("adet" if model_unit == "adet" else None)
    if unit is None:
        raise UnitError(f"unit {unit_text!r} has no canonical unit")
    if model_unit != unit:
        raise FigureError(f"model unit {model_unit!r} != unit of {unit_text!r} ({unit})")
    if model_value is None:
        raise FigureError("model gave no normalised value")
    model = Decimal(str(model_value))
    if abs(model - quantity.value) > VALUE_REL_EPS * max(abs(quantity.value), Decimal(1)):
        raise FigureError(
            f"model value {model_value} != {shown!r} x {unit_text!r} = {quantity.value}"
        )
    value = quantity.value
    if unit in ("TL", "adet"):
        value = value.to_integral_value()
    return to_json_number(value), unit


def _check_scale_on_page(text: str, page_text: str) -> None:
    scales = [w for w in re.findall(r"\w+", fold(text)) if w in SCALE_WORDS]
    if not scales:
        return
    page_words = set(re.findall(r"\w+", fold(page_text)))
    if not page_words & set(scales):
        raise FigureError(f"unit scale {scales[0]!r} is not written on the page")


def rounding_tolerance(
    figure: str | None, unit_text: str | None, unit: str, value: int | float
) -> Tolerance:
    """The spec tolerance for `unit`, widened to cover the rounding of the printed figure.

    A figure printed as "2,6 trilyon TL" only pins the value to +-0.05 trilyon (1.9%), so
    the exact statement figure must still score as correct; "%17" pins a ratio to +-0.5
    points; "27,7 milyon" customers to +-50 000. Exact figures keep the spec tolerance.
    Components (summed parts) are exact counts and keep it too.
    """
    base = TOLERANCE_BY_UNIT[unit]
    if not figure:
        return base
    tokens = NUMBER_TOKEN.findall(figure)
    parts = _COMPOUND_PART.findall(fold(figure))
    try:
        last = parse_number(tokens[-1])
        scale = Decimal(_SCALES[parts[-1][1]]) if len(parts) == len(tokens) > 1 else Decimal(1)
        scale *= normalize(Decimal(1), unit_text or "").value if unit != "%" else Decimal(1)
    except (NumericParseError, IndexError, KeyError):
        return base
    half_unit = Decimal(5) * Decimal(10) ** (last.as_tuple().exponent - 1) * scale
    if unit == "TL":
        if not value:
            return base
        rel = half_unit / abs(Decimal(str(value)))
        amount = max(Decimal(str(base.amount)), _round_up(rel, 4))
        return Tolerance(type="rel", amount=float(amount))
    if half_unit <= Decimal("0.5") and unit == "adet":
        return base
    amount = max(Decimal(str(base.amount)), _round_up(half_unit, 2))
    return Tolerance(type="abs", amount=int(amount) if unit == "adet" else float(amount))


def _round_up(x: Decimal, places: int) -> Decimal:
    q = Decimal(10) ** -places
    return x.quantize(q, rounding=ROUND_CEILING)
