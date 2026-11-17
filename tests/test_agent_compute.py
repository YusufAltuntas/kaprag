"""Deterministic calculation of the router's `compute` node (pure; no LLM)."""

import pytest

from kaprag.agent.compute import Operand, compute, format_quantity


def op(id_, value, unit, basis=None, missing=None, label=None):
    return Operand(
        id=id_, label=label or f"label-{id_}", value=value, unit=unit, basis=basis, missing=missing
    )


# --- percent change -----------------------------------------------------------


def test_percent_change_is_target_over_base_rounded_to_two_places():
    result = compute("percent_change", [op("b", 694, "adet"), op("a", 1034, "adet")])
    assert result.ok
    assert (result.value, result.unit) == (48.99, "%")


def test_percent_change_keeps_the_sign_of_a_decrease():
    result = compute(
        "percent_change",
        [op("y23", 4_393_157_000, "TL", "solo"), op("y24", 3_225_081_000, "TL", "solo")],
    )
    assert (result.value, result.unit, result.basis) == (-26.59, "%", "solo")


def test_percent_change_rounds_half_up():
    # (15.1 - 15.2) / 15.2 * 100 = -0.65789... -> -0.66
    result = compute(
        "percent_change", [op("a", 15.2, "%", "konsolide"), op("b", 15.1, "%", "konsolide")]
    )
    assert result.value == -0.66


def test_percent_change_from_zero_is_refused():
    result = compute(
        "percent_change", [op("a", 0, "TL", "konsolide"), op("b", 5, "TL", "konsolide")]
    )
    assert not result.ok and result.value is None
    assert "sıfır" in result.refusal


# --- difference, ratio, share -------------------------------------------------


def test_difference_of_money_is_integer_tl():
    result = compute(
        "difference",
        [op("g", 86_907_000_000, "TL", "konsolide"), op("y", 68_010_000_000, "TL", "konsolide")],
    )
    assert (result.value, result.unit, result.basis) == (18_897_000_000, "TL", "konsolide")
    assert isinstance(result.value, int)


def test_difference_of_ratios_is_percentage_points():
    result = compute(
        "difference", [op("a", 19.03, "%", "konsolide"), op("y", 16.71, "%", "konsolide")]
    )
    assert (result.value, result.unit) == (2.32, "%")


def test_difference_of_counts_is_a_count():
    result = compute("difference", [op("i", 1034, "adet"), op("a", 694, None)])
    assert (result.value, result.unit) == (340, "adet")


def test_ratio_is_unitless_times_and_share_is_percent():
    ratio = compute("ratio", [op("i", 1034, "adet"), op("a", 694, "adet")])
    assert (ratio.value, ratio.unit) == (1.4899, None)
    share = compute("share", [op("a", 694, "adet"), op("i", 1034, "adet")])
    assert (share.value, share.unit) == (67.12, "%")


def test_ratio_by_zero_is_refused():
    result = compute("ratio", [op("a", 3, "adet"), op("b", 0, "adet")])
    assert not result.ok and "sıfır" in result.refusal


# --- max, min, rank -------------------------------------------------------------


def test_max_names_the_winner_and_carries_its_figure():
    result = compute("max", [op("a", 694, "adet"), op("g", 794, "adet")])
    assert result.ok
    assert result.winners == ("g",)
    assert (result.value, result.unit) == (794, "adet")


def test_min_over_money():
    result = compute(
        "min",
        [
            op("a", 66_496_235_000, "TL", "konsolide"),
            op("g", 86_907_000_000, "TL", "konsolide"),
            op("y", 68_010_000_000, "TL", "konsolide"),
        ],
    )
    assert result.winners == ("a",)
    assert (result.value, result.unit, result.basis) == (66_496_235_000, "TL", "konsolide")


def test_max_with_a_tie_names_every_winner():
    result = compute("max", [op("a", 794, "adet"), op("b", 694, "adet"), op("c", 794, "adet")])
    assert result.winners == ("a", "c")
    assert result.value == 794


def test_rank_orders_by_value_ties_keep_plan_order():
    operands = [
        op("a", 705, "adet"),
        op("g", 804, "adet"),
        op("i", 1088, "adet"),
        op("y", 804, "adet"),
    ]
    desc = compute("rank", operands)
    assert desc.order == ("i", "g", "y", "a")
    assert (desc.value, desc.unit) == (1088, "adet")
    asc = compute("rank", operands, descending=False)
    assert asc.order == ("a", "g", "y", "i")
    assert asc.value == 705


# --- refusals -------------------------------------------------------------------


def test_unit_mismatch_is_refused():
    result = compute(
        "difference", [op("a", 5_000, "TL", "konsolide"), op("b", 12.5, "%", "konsolide")]
    )
    assert not result.ok and result.value is None
    assert "birim" in result.refusal


def test_basis_mismatch_is_refused_for_money_and_ratios():
    result = compute("max", [op("a", 10, "TL", "konsolide"), op("b", 20, "TL", "solo")])
    assert not result.ok and "esas" in result.refusal


def test_unknown_basis_next_to_a_known_one_is_refused_for_money():
    result = compute("percent_change", [op("a", 10, "TL", "konsolide"), op("b", 20, "TL", None)])
    assert not result.ok and "esas" in result.refusal


def test_basis_does_not_matter_for_counts():
    result = compute("max", [op("a", 694, "adet", "konsolide"), op("g", 794, "adet", None)])
    assert result.ok and result.winners == ("g",) and result.basis is None


def test_missing_operand_is_an_abstention_naming_it():
    result = compute(
        "percent_change",
        [
            op("a", None, None, missing="kaynaklarda yok", label="Akbank 2024 şube sayısı"),
            op("b", 794, "adet", label="Garanti BBVA 2024 şube sayısı"),
        ],
    )
    assert not result.ok and result.value is None
    assert result.missing == ("Akbank 2024 şube sayısı",)
    assert result.refusal is None


def test_operand_without_a_value_counts_as_missing_even_without_a_reason():
    result = compute("max", [op("a", None, None, label="A"), op("b", 1, "adet", label="B")])
    assert result.missing == ("A",)


# --- formatting -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "unit", "text"),
    [
        (48.99, "%", "%48,99"),
        (-26.59, "%", "-%26,59"),
        (18_897_000_000, "TL", "18.897.000.000 TL"),
        (1034, "adet", "1.034"),
        (1.4899, None, "1,4899"),
        (2.5, "%", "%2,5"),
    ],
)
def test_format_quantity_in_turkish(value, unit, text):
    assert format_quantity(value, unit) == text
