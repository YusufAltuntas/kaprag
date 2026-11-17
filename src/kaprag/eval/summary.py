"""Aggregate per-question result rows into `summary.json` groups.

Every metric is reported as `{"value": mean or null, "n": questions it applies to}`;
`n == 0` gives `value: null`, never 0. Booleans average as rates. The same block is
computed for the whole selected set, the `verified: true` subset (a human checked the
page), the `audited` subset (an agent audit `passed` or `fixed` the row; NOT human
verification) and each question type (all seven types are always present so runs
line up column by column).

Operational block per group:
- `cost_usd_total` / `cost_usd_per_question`: the pipeline's API-equivalent cost
  (judge calls excluded), cache hits counted at the original call's cost.
- `latency_ms_p50` / `latency_ms_p95`: pipeline wall time of `ok` questions whose
  pipeline LLM calls were all fresh (no cache hit); `latency_n` is how many such
  questions there were, `latency_excluded_cached` / `latency_excluded_error` how
  many were left out and why. Percentiles interpolate linearly between ranks.
- `judge_cost_usd_total`: judge calls only, reported separately.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any

from kaprag.eval.schema import AUDITED_STATUSES, QUESTION_TYPES

# summary metric name -> per-question field in results.jsonl `metrics`
METRICS: dict[str, str] = {
    "answer_accuracy": "answer_score",
    "page_recall@5": "page_recall@5",
    "page_recall@10": "page_recall@10",
    "all_evidence_recall@10": "all_evidence_recall@10",
    "mrr@10": "mrr@10",
    "numeric_accuracy": "numeric_correct",
    "text_correctness": "text_correct",
    "key_point_coverage": "key_point_coverage",
    "citation_precision": "citation_precision",
    "citation_recall": "citation_recall",
    "behavior_accuracy": "behavior_correct",
    "correct_abstain_rate": "correct_abstain",
    "false_abstain_rate": "false_abstain",
    "clarify_accuracy": "clarify_correct",
    "assumption_accuracy": "assumption_correct",
}


def mean_metric(values: Iterable[Any]) -> dict[str, Any]:
    present = [float(v) for v in values if v is not None]
    return {"value": sum(present) / len(present) if present else None, "n": len(present)}


def percentile(values: Sequence[float], q: float) -> float | None:
    """Linear-interpolated percentile (numpy's default method); None for no values."""
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def _round(value: float | None, digits: int) -> float | None:
    return None if value is None else round(value, digits)


def summarize_group(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    n_error = sum(r["status"] == "error" for r in rows)
    metrics = {
        name: mean_metric(r["metrics"][field] for r in rows) for name, field in METRICS.items()
    }

    eligible = [r["latency_ms"] for r in rows if r["status"] == "ok" and r["latency_eligible"]]
    cost = sum(r["cost_usd"] for r in rows)
    operational = {
        "cost_usd_total": round(cost, 6) if n else None,
        "cost_usd_per_question": round(cost / n, 6) if n else None,
        "input_tokens_total": sum(r["input_tokens"] for r in rows),
        "output_tokens_total": sum(r["output_tokens"] for r in rows),
        "llm_calls_total": sum(r["llm_calls"] for r in rows),
        "cached_llm_calls_total": sum(r["cached_llm_calls"] for r in rows),
        "latency_ms_p50": _round(percentile(eligible, 50), 1),
        "latency_ms_p95": _round(percentile(eligible, 95), 1),
        "latency_n": len(eligible),
        "latency_excluded_cached": sum(
            r["status"] == "ok" and not r["latency_eligible"] for r in rows
        ),
        "latency_excluded_error": n_error,
        "judge_cost_usd_total": round(sum(r["judge_cost_usd"] for r in rows), 6) if n else None,
        "judge_errors": sum(r["judge_error"] is not None for r in rows),
    }
    return {
        "n": n,
        "n_ok": n - n_error,
        "n_error": n_error,
        "metrics": metrics,
        "operational": operational,
    }


def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """`overall`, `verified`, `audited` and `by_type` groups."""
    return {
        "overall": summarize_group(rows),
        "verified": summarize_group([r for r in rows if r["verified"]]),
        "audited": summarize_group([r for r in rows if r["audit"] in AUDITED_STATUSES]),
        "by_type": {
            t: summarize_group([r for r in rows if r["type"] == t]) for t in QUESTION_TYPES
        },
    }
