"""Decomposition plan schema, its validation, and the report-year rule."""

import json

import pytest
from pydantic import ValidationError

from kaprag.agent.plan import (
    MAX_LOOKUPS,
    OPERATIONS,
    Plan,
    ReportYear,
    apply_report_year_rule,
    decompose_system_prompt,
)
from kaprag.corpus import BANK_NAMES
from kaprag.rag.query_filter import CORPUS_YEARS


def lookup(id_, bank="akbank", fiscal_year=2024, report_year=2024, **extra):
    return {
        "id": id_,
        "bank": bank,
        "fiscal_year": fiscal_year,
        "report_year": report_year,
        "metric": "şube sayısı",
        "sub_question": f"{bank} {fiscal_year} şube sayısı?",
        **extra,
    }


def plan(lookups, kind="max", operands=None, **extra):
    return {
        "expressible": True,
        "lookups": lookups,
        "operation": {"kind": kind, "operands": operands or [x["id"] for x in lookups]},
        **extra,
    }


def parse(data):
    return Plan.model_validate_json(json.dumps(data))


def test_closed_operation_set():
    assert set(OPERATIONS) == {
        "percent_change",
        "difference",
        "ratio",
        "share",
        "max",
        "min",
        "rank",
    }
    with pytest.raises(ValidationError, match="kind"):
        parse(plan([lookup("a"), lookup("b", "garanti")], kind="average"))


def test_valid_plan_parses():
    p = parse(plan([lookup("a"), lookup("b", "garanti")], kind="percent_change"))
    assert p.operation.kind == "percent_change"
    assert [x.bank for x in p.lookups] == ["akbank", "garanti"]


def test_report_years_are_the_corpus_years():
    assert ReportYear.__args__ == CORPUS_YEARS


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (plan([lookup("a", bank="ziraat"), lookup("b")]), "bank"),
        (plan([lookup("a", report_year=2022, fiscal_year=2022), lookup("b")]), "report_year"),
        (plan([lookup("a", fiscal_year=2025, report_year=2024), lookup("b")]), "after report_year"),
        (plan([lookup("a", fiscal_year="2024"), lookup("b")]), "fiscal_year"),
        (
            plan([lookup("a"), lookup("a", "garanti"), lookup("b")], operands=["a", "b"]),
            "ids repeat",
        ),
        (plan([lookup("a"), lookup("b")], operands=["a", "c"]), "not look-up ids"),
        (plan([lookup("a"), lookup("b"), lookup("c")], operands=["a", "b"]), "not used"),
        (plan([lookup("a"), lookup("b"), lookup("c")], kind="difference"), "exactly 2"),
        (plan([lookup("a")], kind="max"), "operands"),
        (plan([lookup("a"), lookup("b")], operands=["a", "a"]), "repeat"),
        ({"expressible": True, "lookups": [lookup("a"), lookup("b")]}, "needs an operation"),
    ],
)
def test_invalid_plans_are_rejected(data, message):
    with pytest.raises(ValidationError, match=message):
        parse(data)


def test_lookup_cap():
    ids = [f"l{i}" for i in range(MAX_LOOKUPS + 1)]
    assert MAX_LOOKUPS == 8
    parse(plan([lookup(i) for i in ids[:MAX_LOOKUPS]]))
    with pytest.raises(ValidationError, match="lookups"):
        parse(plan([lookup(i) for i in ids]))


def test_inexpressible_plan_needs_nothing_else():
    p = parse({"expressible": False, "reason": "ortalama işlemi yok"})
    assert not p.expressible and p.lookups == [] and p.operation is None


def test_bank_keys_match_the_corpus():
    from kaprag.agent.plan import Lookup

    assert set(Lookup.model_fields["bank"].annotation.__args__) == set(BANK_NAMES)


def test_year_rule_moves_both_figures_of_a_change_to_the_later_report():
    p = parse(
        plan(
            [
                lookup("y24", "yapikredi", fiscal_year=2024, report_year=2024),
                lookup("y25", "yapikredi", fiscal_year=2025, report_year=2025),
            ],
            kind="percent_change",
        )
    )
    fixed, fixes = apply_report_year_rule(p)
    assert [x.report_year for x in fixed.lookups] == [2025, 2025]
    assert fixes == [{"id": "y24", "from": 2024, "to": 2025}]


def test_year_rule_reads_2022_from_the_2023_report():
    p = parse(
        plan(
            [
                lookup("g22", "garanti", fiscal_year=2022, report_year=2023),
                lookup("g23", "garanti", fiscal_year=2023, report_year=2023),
            ],
            kind="percent_change",
        )
    )
    fixed, fixes = apply_report_year_rule(p)
    assert fixes == [] and fixed == p


@pytest.mark.parametrize(
    "lookups",
    [
        # two banks, same year: each from its own year's report
        [lookup("a", "akbank"), lookup("g", "garanti")],
        # same bank, two years apart: the later report has no such comparative column
        [
            lookup("a", fiscal_year=2023, report_year=2023),
            lookup("b", fiscal_year=2025, report_year=2025),
        ],
    ],
)
def test_year_rule_leaves_other_binary_plans_alone(lookups):
    p = parse(plan(lookups, kind="difference"))
    assert apply_report_year_rule(p) == (p, [])


def test_year_rule_ignores_selection_operations():
    p = parse(
        plan(
            [
                lookup("a", fiscal_year=2024, report_year=2024),
                lookup("b", fiscal_year=2025, report_year=2025),
            ],
            kind="max",
        )
    )
    assert apply_report_year_rule(p) == (p, [])


def test_decompose_prompt_states_the_rules():
    prompt = decompose_system_prompt(6)
    assert "en fazla 6 tane" in prompt
    assert "karşılaştırmalı sütundan" in prompt  # section 4: the later report's comparative column
    for kind in OPERATIONS:
        assert f"`{kind}`" in prompt
    assert "{" not in prompt
