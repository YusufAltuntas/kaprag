"""Turkish number parsing, unit normalisation and tolerance comparison (pure, no I/O).

Canonical units (eval/README.md §3): money in integer **TL** with the bin/milyon/milyar
multiplier applied, ratios in **%**, counts as integers (unit **adet**). A bare number
has unit `None`.

Number format rules (`parse_number`):
- Turkish first: `.` groups thousands, `,` is the decimal mark (`1.234.567,8`).
- When both marks occur, the last one is the decimal mark, so English `1,234,567.8`
  also parses. A single `,` is always decimal (`1,234` == 1.234). A single `.` followed
  by exactly three digits is a thousands separator (`1.234` == 1234) unless the integer
  part is `0` (`0.125`); any other single `.` is decimal (`12.5`).
- Groups of three digits may also be separated by spaces / no-break spaces.
- Negative: leading `-`, `+`/`-`/U+2212 sign, or parentheses as in financial tables.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

CANONICAL_UNITS = ("TL", "%", "adet")
TOLERANCE_TYPES = ("rel", "abs")


class NumericParseError(ValueError):
    """Text or unit that cannot be read as a number / known unit."""


@dataclass(frozen=True)
class Quantity:
    """A number in canonical units (`unit` is "TL", "%", "adet" or None)."""

    value: Decimal
    unit: str | None


@dataclass(frozen=True)
class NumericResult:
    """Outcome of comparing an answer's value/unit with the gold value/unit."""

    correct: bool
    reason: str | None  # None when correct
    value: int | float | None  # the answer's value in canonical units, if parseable
    unit: str | None


# --- numbers ---------------------------------------------------------------

_SPACES = "    "
_MINUS = "-−"
_SPACE_GROUPED = re.compile(r"^\d{1,3}(?:[    ]\d{3})+(?:,\d+)?$")
_TR_INT = re.compile(r"^\d{1,3}(?:\.\d{3})*$")
_EN_INT = re.compile(r"^\d{1,3}(?:,\d{3})*$")
_DIGITS = re.compile(r"^\d+$")


def parse_number(text: str) -> Decimal:
    """Parse one number written in Turkish (or unambiguous English) format."""
    s = text.strip()
    negative = False
    if s.startswith("(") or s.endswith(")"):
        if not (s.startswith("(") and s.endswith(")")):
            raise NumericParseError(f"unbalanced parentheses in {text!r}")
        s = s[1:-1].strip()
        negative = True
    if s and (s[0] in _MINUS or s[0] == "+"):
        negative = negative != (s[0] in _MINUS)
        s = s[1:].lstrip()
    if not s or not s[0].isdigit():
        raise NumericParseError(f"not a number: {text!r}")

    if any(c in s for c in _SPACES):
        if not _SPACE_GROUPED.match(s):
            raise NumericParseError(f"bad digit grouping in {text!r}")
        s = "".join(c for c in s if c not in _SPACES)

    plain = _to_plain(s, text)
    try:
        value = Decimal(plain)
    except InvalidOperation:
        raise NumericParseError(f"not a number: {text!r}") from None
    return -value if negative else value


def _to_plain(s: str, original: str) -> str:
    """Rewrite a separator-formatted number as `1234.5`."""
    has_dot, has_comma = "." in s, "," in s
    if has_dot and has_comma:
        if s.rfind(",") > s.rfind("."):  # Turkish: 1.234,5
            int_part, frac = s.rsplit(",", 1)
            ok = _TR_INT.match(int_part) and _DIGITS.match(frac)
            plain = int_part.replace(".", "") + "." + frac
        else:  # English: 1,234.5
            int_part, frac = s.rsplit(".", 1)
            ok = _EN_INT.match(int_part) and _DIGITS.match(frac)
            plain = int_part.replace(",", "") + "." + frac
    elif has_comma:
        if s.count(",") == 1:  # decimal comma
            int_part, frac = s.split(",")
            ok = _DIGITS.match(int_part) and _DIGITS.match(frac)
            plain = f"{int_part}.{frac}"
        else:  # English thousands: 1,234,567
            ok = _EN_INT.match(s)
            plain = s.replace(",", "")
    elif has_dot:
        int_part, _, frac = s.partition(".")
        if s.count(".") > 1 or (len(frac) == 3 and int_part != "0" and len(int_part) <= 3):
            ok = _TR_INT.match(s)  # Turkish thousands: 1.234 / 1.234.567
            plain = s.replace(".", "")
        else:  # decimal point: 12.5, 0.125
            ok = _DIGITS.match(int_part) and _DIGITS.match(frac)
            plain = s
    else:
        ok = _DIGITS.match(s)
        plain = s
    if not ok:
        raise NumericParseError(f"bad number format: {original!r}")
    return plain


# --- units -----------------------------------------------------------------

# Unit words are matched on a folded key (see `_key`): lower case, Turkish letters
# mapped to ASCII, so "MILYAR", "MİLYAR", "çalışan" and "calisan" all match.
_ASCII = str.maketrans({"ı": "i", "ş": "s", "ç": "c", "ğ": "g", "ö": "o", "ü": "u", "̇": ""})
# Punctuation around unit words, e.g. "(Bin TL)" column headers or "milyar TL.".
_PUNCTUATION = str.maketrans(dict.fromkeys("()[]{},.;:'\"", " "))

_SCALE, _CURRENCY, _COUNT, _PERCENT, _BPS = "scale", "TL", "adet", "%", "bps"
_UNIT_ENTRIES: tuple[tuple[str, str, Decimal], ...] = (
    ("bin", _SCALE, Decimal(10) ** 3),
    ("milyon", _SCALE, Decimal(10) ** 6),
    ("mn", _SCALE, Decimal(10) ** 6),
    ("mio", _SCALE, Decimal(10) ** 6),
    ("milyar", _SCALE, Decimal(10) ** 9),
    ("mlr", _SCALE, Decimal(10) ** 9),
    ("trilyon", _SCALE, Decimal(10) ** 12),
    ("trn", _SCALE, Decimal(10) ** 12),
    ("tl", _CURRENCY, Decimal(1)),
    ("try", _CURRENCY, Decimal(1)),
    ("adet", _COUNT, Decimal(1)),
    ("tane", _COUNT, Decimal(1)),
    ("sube", _COUNT, Decimal(1)),
    ("kisi", _COUNT, Decimal(1)),
    ("calisan", _COUNT, Decimal(1)),
    ("personel", _COUNT, Decimal(1)),
    ("musteri", _COUNT, Decimal(1)),
    ("atm", _COUNT, Decimal(1)),
    ("yuzde", _PERCENT, Decimal(1)),
    ("puan", _PERCENT, Decimal(1)),  # percentage points
    ("bp", _BPS, Decimal("0.01")),  # basis points -> %
    ("bps", _BPS, Decimal("0.01")),
)


def _build_vocabulary() -> dict[str, tuple[str, Decimal]]:
    vocabulary: dict[str, tuple[str, Decimal]] = {}
    for word, kind, multiplier in _UNIT_ENTRIES:
        if word in vocabulary:
            raise AssertionError(f"duplicate unit word {word!r} after folding")
        vocabulary[word] = (kind, multiplier)
    return vocabulary


_UNIT_VOCABULARY = _build_vocabulary()
# Multi-word phrases, on folded keys, rewritten to one vocabulary word.
_PHRASES = (
    (re.compile(r"\bturk lirasi\b"), "tl"),
    (re.compile(r"\bbaz puan\b"), "bp"),
    (re.compile(r"\byuzde puan\b"), "puan"),
)


def _fold(text: str) -> str:
    """Lower-case with Turkish dotted/dotless I handled."""
    return text.replace("İ", "i").replace("I", "ı").lower()


def _key(text: str) -> str:
    """Folded, ASCII-only form used to match unit words: `MILYAR`/`MİLYAR` -> `milyar`."""
    return _fold(text).translate(_ASCII)


def _unit_from_words(
    words: list[str], percent_sign: bool, original: str
) -> tuple[Decimal, str | None]:
    """Return (multiplier, canonical unit) for the (folded) unit words around a number."""
    multiplier = Decimal(1)
    kinds: set[str] = {"%"} if percent_sign else set()
    scaled = False
    for word in words:
        if word not in _UNIT_VOCABULARY:
            raise NumericParseError(f"unknown unit {word!r} in {original!r}")
        kind, factor = _UNIT_VOCABULARY[word]
        if kind == _SCALE:
            if scaled:
                raise NumericParseError(f"more than one scale word in {original!r}")
            scaled = True
        else:
            kinds.add("%" if kind == _BPS else kind)
        multiplier *= factor
    if len(kinds) > 1:
        raise NumericParseError(f"conflicting units {sorted(kinds)} in {original!r}")
    return multiplier, (kinds.pop() if kinds else None)


def _unit_words(text: str) -> tuple[list[str], bool]:
    """Split unit text into folded words; also report whether a % sign was present.

    Surrounding punctuation is dropped, so `(Bin TL)`, `milyar TL.` and `TL, milyar`
    read as plain unit words.
    """
    s = _key(text).replace("₺", " tl ").translate(_PUNCTUATION)
    percent_sign = "%" in s
    s = " ".join(s.replace("%", " ").split())
    for pattern, token in _PHRASES:
        s = pattern.sub(token, s)
    return s.split(), percent_sign


_NUMBER_IN_TEXT = re.compile(
    r"\(?\s*[-+−]?\s*"
    r"(?:\d{1,3}(?:[    ]\d{3})+(?:,\d+)?(?!\d)|\d[\d.,]*\d|\d)"
    r"\s*\)?"
)


def parse_quantity(text: str) -> Quantity:
    """Parse e.g. `45,2 milyar TL`, `%12,5`, `(1.234) TL`, `12 şube` into canonical units."""
    matches = list(_NUMBER_IN_TEXT.finditer(text))
    if len(matches) != 1:
        raise NumericParseError(f"expected exactly one number in {text!r}")
    match = matches[0]
    number = match.group(0).strip()
    rest = text[: match.start()] + " " + text[match.end() :]
    # a sign written before a % sign: "-%3,2"
    rest_stripped = rest.strip()
    if rest_stripped[:1] in _MINUS and "%" in rest_stripped[1:2]:
        number = "-" + number
        rest = rest.replace(rest_stripped[0], "", 1)
    words, percent_sign = _unit_words(rest)
    multiplier, unit = _unit_from_words(words, percent_sign, text)
    return Quantity(parse_number(number) * multiplier, unit)


def _to_decimal(value: object) -> Decimal:
    if isinstance(value, bool) or value is None:
        raise NumericParseError(f"not a number: {value!r}")
    if isinstance(value, Decimal):
        result = value
    elif isinstance(value, int):
        result = Decimal(value)
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise NumericParseError(f"not a finite number: {value!r}")
        result = Decimal(repr(value))  # shortest repr: 0.1 -> Decimal("0.1")
    elif isinstance(value, str):
        result = parse_number(value)
    else:
        raise NumericParseError(f"not a number: {value!r}")
    if not result.is_finite():
        raise NumericParseError(f"not a finite number: {value!r}")
    return result


def normalize(value: object, unit: str | None) -> Quantity:
    """Normalise a structured answer's `value` + free-text `unit` to canonical units.

    `unit` may carry a scale word (`milyar TL`, `bin TL`) which is applied to `value`.
    Raises NumericParseError for an unknown unit or a non-numeric value.
    """
    number = _to_decimal(value)
    if unit is None or not unit.strip():
        return Quantity(number, None)
    if any(c.isdigit() for c in unit):
        raise NumericParseError(f"unit contains digits: {unit!r}")
    words, percent_sign = _unit_words(unit)
    multiplier, canonical = _unit_from_words(words, percent_sign, unit)
    return Quantity(number * multiplier, canonical)


# --- comparison ------------------------------------------------------------


def within_tolerance(predicted: object, expected: object, kind: str, amount: float) -> bool:
    """`rel`: |p - e| <= amount * |e|; `abs`: |p - e| <= amount. Both inclusive."""
    p, e, a = _to_decimal(predicted), _to_decimal(expected), _to_decimal(amount)
    if kind == "rel":
        return abs(p - e) <= a * abs(e)
    if kind == "abs":
        return abs(p - e) <= a
    raise ValueError(f"unknown tolerance type {kind!r}; expected one of {TOLERANCE_TYPES}")


def to_json_number(value: Decimal) -> int | float:
    """Integral values as int (money stays exact), others as float."""
    return int(value) if value == value.to_integral_value() else float(value)


def compare(
    answer_value: object,
    answer_unit: str | None,
    expected_value: object,
    expected_unit: str | None,
    tolerance_type: str,
    tolerance_amount: float,
) -> NumericResult:
    """Score an answer's value/unit against the gold value/unit with a tolerance.

    Units must match after normalisation; a unit-less answer matches only `adet`
    (counts are unit-less numbers). Never raises for a bad answer: an unparseable
    value or unit is scored as wrong with the reason.
    """
    if answer_value is None:
        return NumericResult(False, "no value in answer", None, None)
    try:
        got = normalize(answer_value, answer_unit)
    except NumericParseError as exc:
        return NumericResult(False, f"unparseable answer: {exc}", None, None)
    want = normalize(expected_value, expected_unit)
    got_unit = got.unit if got.unit is not None or want.unit != "adet" else "adet"
    value = to_json_number(got.value)
    if got_unit != want.unit:
        reason = f"unit mismatch: {got.unit!r} != {want.unit!r}"
        return NumericResult(False, reason, value, got.unit)
    if not within_tolerance(got.value, want.value, tolerance_type, tolerance_amount):
        reason = (
            f"outside tolerance: {value} vs {to_json_number(want.value)} "
            f"({tolerance_type} {tolerance_amount})"
        )
        return NumericResult(False, reason, value, got_unit)
    return NumericResult(True, None, value, got_unit)
