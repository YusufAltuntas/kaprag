"""Distribution report of a gold file (the T6 acceptance view)."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from kaprag.eval.schema import BEHAVIORS, QUESTION_TYPES, Question
from kaprag.eval.split import stratum
from kaprag.gold.assemble import SPEC_TARGETS

BANK_COLUMNS = ("garanti", "akbank", "isbank", "yapikredi", "multi", "none")


def _table(header: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    cells = [list(map(str, header))] + [list(map(str, r)) for r in rows]
    widths = [max(len(r[i]) for r in cells) for i in range(len(header))]
    lines = ["  ".join(c.ljust(w) for c, w in zip(r, widths, strict=True)) for r in cells]
    lines.insert(1, "  ".join("-" * w for w in widths))
    return "\n".join(lines)


def distribution_report(questions: Sequence[Question]) -> str:
    """Type x split x bank table, totals vs spec targets, pilot, behaviour and check counts."""
    out: list[str] = [f"questions: {len(questions)}", ""]

    cells = Counter((q.type, q.split, stratum(q)[1]) for q in questions)
    rows = []
    for type_ in QUESTION_TYPES:
        for split in ("dev", "test"):
            counts = [cells[(type_, split, b)] for b in BANK_COLUMNS]
            rows.append([type_, split, *counts, sum(counts)])
    out += ["type x split x bank", _table(["type", "split", *BANK_COLUMNS, "total"], rows), ""]

    years = Counter((q.type, y) for q in questions for y in (q.years or [0]))
    year_cols = sorted({y for _, y in years})
    rows = [[t, *[years[(t, y)] for y in year_cols]] for t in QUESTION_TYPES]
    labels = [str(y) if y else "none" for y in year_cols]
    out += [
        "type x year (a question counts once per year it names)",
        _table(["type", *labels], rows),
        "",
    ]

    by_type = Counter(q.type for q in questions)
    split_by_type = Counter((q.type, q.split) for q in questions)
    rows = []
    for t in QUESTION_TYPES:
        target = SPEC_TARGETS[t]
        ratio = by_type[t] / target
        status = "ok" if 1.0 <= ratio <= 1.6 else ("BELOW" if ratio < 1 else "ABOVE 1.6x")
        rows.append(
            [
                t,
                by_type[t],
                target,
                f"{ratio:.2f}",
                status,
                split_by_type[(t, "dev")],
                split_by_type[(t, "test")],
            ]
        )
    rows.append(
        [
            "total",
            len(questions),
            sum(SPEC_TARGETS.values()),
            "",
            "",
            sum(q.split == "dev" for q in questions),
            sum(q.split == "test" for q in questions),
        ]
    )
    out += [
        "per type vs spec target (eval/README.md §1)",
        _table(["type", "count", "target", "ratio", "status", "dev", "test"], rows),
        "",
    ]

    pilot = Counter(q.type for q in questions if q.pilot)
    pilot_rows = [[t, pilot[t]] for t in QUESTION_TYPES] + [["total", sum(pilot.values())]]
    pilot_test = sum(q.pilot and q.split == "test" for q in questions)
    out += [
        f"pilot by type (pilot questions in test split: {pilot_test})",
        _table(["type", "pilot"], pilot_rows),
        "",
    ]

    behaviors = Counter(q.expected_behavior for q in questions)
    out += [
        "expected_behavior",
        _table(["behavior", "count"], [[b, behaviors[b]] for b in BEHAVIORS]),
        "",
    ]

    authors = Counter(q.author for q in questions)
    checked = sum(q.llm_checked for q in questions)
    derived = sum(bool(q.derived_from) for q in questions)
    verified = sum(q.verified for q in questions)
    out += [
        f"author: {dict(sorted(authors.items()))}",
        f"llm_checked: {checked} true, {len(questions) - checked} false",
        f"derived (derived_from non-empty): {derived}",
        f"verified: true = {verified}",
    ]
    return "\n".join(out)
