"""Turkish number parsing, unit normalisation and tolerance comparison."""

from decimal import Decimal

import pytest

from kaprag.eval.numeric import (
    NumericParseError,
    Quantity,
    compare,
    normalize,
    parse_number,
    parse_quantity,
    within_tolerance,
)

# --- parse_number ----------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("1.234.567,8", "1234567.8"),
        ("12,5", "12.5"),
        ("0,25", "0.25"),
        ("1.234", "1234"),
        ("1234", "1234"),
        ("45", "45"),
        (" 45 ", "45"),
        ("-1.234,5", "-1234.5"),
        ("−5", "-5"),  # unicode minus sign
        ("(1.234)", "-1234"),
        ("(1.234,5)", "-1234.5"),
        ("+3,5", "3.5"),
        # English formats an LLM may still produce
        ("1,234,567.8", "1234567.8"),
        ("12.5", "12.5"),
        ("0.125", "0.125"),
        ("1,234,567", "1234567"),
        # space / no-break-space grouping
        ("1 234 567", "1234567"),
        ("1 234,5", "1234.5"),
    ],
)
def test_parse_number(text, expected):
    assert parse_number(text) == Decimal(expected)


@pytest.mark.parametrize(
    "text", ["", "   ", "abc", "1.23.456", "12,34,5", "--5", "1,2,3", "(12", "12)", "1.2.3,4.5"]
)
def test_parse_number_rejects_garbage(text):
    with pytest.raises(NumericParseError):
        parse_number(text)


# --- parse_quantity --------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "value", "unit"),
    [
        ("%12,5", "12.5", "%"),
        ("12,5%", "12.5", "%"),
        ("% 12,5", "12.5", "%"),
        ("12,5 %", "12.5", "%"),
        ("yüzde 12,5", "12.5", "%"),
        ("Yüzde 12,5", "12.5", "%"),
        ("-%3,2", "-3.2", "%"),
        ("%-3,2", "-3.2", "%"),
        ("45,2 milyar TL", "45200000000", "TL"),
        ("45,2 Milyar TL", "45200000000", "TL"),
        ("45,2 MİLYAR TL", "45200000000", "TL"),
        ("1,2 trilyon TL", "1200000000000", "TL"),
        ("350 bin TL", "350000", "TL"),
        ("12,3 milyon TL", "12300000", "TL"),
        ("12,3 mn TL", "12300000", "TL"),
        ("7,5 mlr TL", "7500000000", "TL"),
        ("1.234.567 TL", "1234567", "TL"),
        ("1.234.567 ₺", "1234567", "TL"),
        ("₺1.234.567", "1234567", "TL"),
        ("TL 1.234", "1234", "TL"),
        ("1.234 TRY", "1234", "TL"),
        ("-45,2 milyar TL", "-45200000000", "TL"),
        ("(45,2) milyar TL", "-45200000000", "TL"),
        ("(1.234.567) TL", "-1234567", "TL"),
        ("1,2 trilyon", "1200000000000", None),
        ("45,2 milyar", "45200000000", None),
        ("1.234", "1234", None),
        ("12 şube", "12", "adet"),
        ("1.024 adet", "1024", "adet"),
        ("1,2 milyon müşteri", "1200000", "adet"),
        ("18.500 çalışan", "18500", "adet"),
        ("2,3 puan", "2.3", "%"),
        ("2,3 yüzde puan", "2.3", "%"),
        ("150 baz puan", "1.5", "%"),
        ("150 bp", "1.5", "%"),
    ],
)
def test_parse_quantity(text, value, unit):
    assert parse_quantity(text) == Quantity(Decimal(value), unit)


@pytest.mark.parametrize("text", ["", "milyar TL", "12 USD", "12 elma", "%", "12 milyar milyon TL"])
def test_parse_quantity_rejects(text):
    with pytest.raises(NumericParseError):
        parse_quantity(text)


# --- normalize (structured answer value + unit) -----------------------------


@pytest.mark.parametrize(
    ("value", "unit", "exp_value", "exp_unit"),
    [
        (45.2, "milyar TL", "45200000000", "TL"),
        (45200000000, "TL", "45200000000", "TL"),
        (1234, "₺", "1234", "TL"),
        (350, "bin TL", "350000", "TL"),
        (12.5, "%", "12.5", "%"),
        (12.5, "yüzde", "12.5", "%"),
        (5, None, "5", None),
        (5, "", "5", None),
        (1024, "adet", "1024", "adet"),
        (1.2, "milyon müşteri", "1200000", "adet"),
        ("1.234,5", "milyon TL", "1234500000", "TL"),
        (0.1, "milyar TL", "100000000", "TL"),  # no float artefacts
    ],
)
def test_normalize(value, unit, exp_value, exp_unit):
    assert normalize(value, unit) == Quantity(Decimal(exp_value), exp_unit)


@pytest.mark.parametrize("unit", ["USD", "elma", "milyar"])
def test_normalize_rejects_unknown_or_scale_only_units(unit):
    if unit == "milyar":
        # a bare scale word is a valid unit: 45,2 milyar -> number without unit
        assert normalize(45.2, unit) == Quantity(Decimal("45200000000"), None)
        return
    with pytest.raises(NumericParseError):
        normalize(3.5, unit)


@pytest.mark.parametrize(
    ("unit", "multiplier", "canonical"),
    [
        # financial-statement headers, punctuation, Turkish/ASCII capital I
        ("(Bin TL)", "1000", "TL"),
        ("(bin TL)", "1000", "TL"),
        ("[Bin TL]", "1000", "TL"),
        ("Bin TL", "1000", "TL"),
        ("BİN TL", "1000", "TL"),
        ("BIN TL", "1000", "TL"),
        ("milyar TL.", "1000000000", "TL"),
        ("TL, milyar", "1000000000", "TL"),
        ("MILYAR TL", "1000000000", "TL"),
        ("MİLYAR TL", "1000000000", "TL"),
        ("milyon ₺", "1000000", "TL"),
        ("Milyon TL", "1000000", "TL"),
        ("milyon TL;", "1000000", "TL"),
        ('"TL"', "1", "TL"),
        ("TÜRK LİRASI", "1", "TL"),
        ("turk lirasi", "1", "TL"),
        ("%", "1", "%"),
        ("(%)", "1", "%"),
        ("yüzde", "1", "%"),
        ("YÜZDE", "1", "%"),
        ("yuzde", "1", "%"),
        ("Yüzde Puan", "1", "%"),
        ("Baz Puan", "0.01", "%"),
        # count nouns: dotted / dotless / ASCII spellings
        ("şube", "1", "adet"),
        ("ŞUBE", "1", "adet"),
        ("sube", "1", "adet"),
        ("kişi", "1", "adet"),
        ("KİŞİ", "1", "adet"),
        ("KISI", "1", "adet"),
        ("kisi", "1", "adet"),
        ("çalışan", "1", "adet"),
        ("ÇALIŞAN", "1", "adet"),
        ("calisan", "1", "adet"),
        ("CALISAN", "1", "adet"),
        ("müşteri", "1", "adet"),
        ("MÜŞTERİ", "1", "adet"),
        ("musteri", "1", "adet"),
        ("ADET", "1", "adet"),
        ("Tane", "1", "adet"),
        ("Personel", "1", "adet"),
        ("ATM", "1", "adet"),
        ("milyon müşteri", "1000000", "adet"),
    ],
)
def test_normalize_accepts_unit_spellings(unit, multiplier, canonical):
    assert normalize(2, unit) == Quantity(2 * Decimal(multiplier), canonical)


@pytest.mark.parametrize(
    "text", ["5 (Bin TL)", "5 bin TL.", "5 BIN TL", "5 MILYAR TL", "12 ŞUBE", "12 calisan"]
)
def test_parse_quantity_accepts_unit_spellings(text):
    assert parse_quantity(text).unit in ("TL", "adet")


@pytest.mark.parametrize(
    "unit",
    [
        "USD",
        "elma",
        "(USD)",
        "dolar.",
        "milyar dolar",
        "TL USD",
        "milyar milyon TL",
        "bin, milyon",
        "TL %",
        "şube TL",
        "mılyon_TL",
        "binTL",
    ],
)
def test_unknown_or_conflicting_units_still_fail(unit):
    with pytest.raises(NumericParseError):
        normalize(3, unit)


def test_unit_vocabulary_has_no_collisions_after_folding():
    from kaprag.eval.numeric import _UNIT_VOCABULARY

    # every folded key maps to exactly one meaning (dict construction would hide dups)
    assert len(_UNIT_VOCABULARY) == len(set(_UNIT_VOCABULARY))
    assert {"bin", "milyon", "milyar", "trilyon", "tl", "adet", "yuzde", "puan"} <= set(
        _UNIT_VOCABULARY
    )


def test_compare_folds_units_on_both_sides():
    assert compare(5, "(Bin TL)", 5000, "TL", "rel", 0.01).correct
    assert compare(5, "MILYAR TL", 5_000_000_000, "TL", "rel", 0.01).correct
    assert compare(12, "ŞUBE", 12, "adet", "abs", 0).correct
    assert compare(12.5, "(%)", 12.5, "%", "abs", 0.2).correct
    assert not compare(5, "(Bin TL)", 5000, "%", "abs", 0.2).correct


def test_normalize_rejects_bool_and_nan():
    with pytest.raises(NumericParseError):
        normalize(True, "TL")
    with pytest.raises(NumericParseError):
        normalize(float("nan"), "TL")


# --- tolerance -------------------------------------------------------------


@pytest.mark.parametrize(
    ("predicted", "expected", "kind", "amount", "ok"),
    [
        (45_200_000_000, 45_213_456_789, "rel", 0.01, True),
        (46_000_000_000, 45_200_000_000, "rel", 0.01, False),
        (101, 100, "rel", 0.01, True),  # boundary is inclusive
        (102, 100, "rel", 0.01, False),
        (-101, -100, "rel", 0.01, True),  # relative to |expected|
        (0, 0, "rel", 0.01, True),
        (1, 0, "rel", 0.01, False),
        (12.7, 12.5, "abs", 0.2, True),  # exact decimal boundary, no float artefact
        (12.3, 12.5, "abs", 0.2, True),
        (12.75, 12.5, "abs", 0.2, False),
        (12, 12, "abs", 0, True),
        (13, 12, "abs", 0, False),
    ],
)
def test_within_tolerance(predicted, expected, kind, amount, ok):
    assert within_tolerance(predicted, expected, kind, amount) is ok


def test_within_tolerance_rejects_unknown_kind():
    with pytest.raises(ValueError):
        within_tolerance(1, 1, "pct", 0.1)


# --- compare (full numeric scorer) -----------------------------------------


def test_compare_money_with_scale_word():
    result = compare(45.2, "milyar TL", 45_213_456_789, "TL", "rel", 0.01)
    assert result.correct
    assert result.value == 45_200_000_000
    assert result.unit == "TL"
    assert result.reason is None


def test_compare_unit_mismatch_is_wrong():
    result = compare(12.5, "%", 12.5, "TL", "rel", 0.01)
    assert not result.correct
    assert "unit" in result.reason


def test_compare_missing_value_is_wrong():
    result = compare(None, "TL", 100, "TL", "rel", 0.01)
    assert not result.correct
    assert result.reason == "no value in answer"


def test_compare_unparseable_unit_is_wrong_not_raised():
    result = compare(3.5, "USD", 100, "TL", "rel", 0.01)
    assert not result.correct
    assert "USD" in result.reason


def test_compare_count_accepts_missing_unit():
    assert compare(12, None, 12, "adet", "abs", 0).correct
    assert compare(12, "şube", 12, "adet", "abs", 0).correct
    assert not compare(13, None, 12, "adet", "abs", 0).correct


def test_compare_out_of_tolerance_reports_reason():
    result = compare(13.0, "%", 12.5, "%", "abs", 0.2)
    assert not result.correct
    assert "tolerance" in result.reason
