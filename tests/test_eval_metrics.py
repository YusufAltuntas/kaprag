"""Retrieval, citation, behaviour and headline metrics; summary aggregation."""

import json
from pathlib import Path

import pytest

from kaprag.eval.answer import Citation, SystemAnswer
from kaprag.eval.metrics import (
    all_evidence_recall_at_k,
    behavior_correct,
    citation_scores,
    content_metric,
    mrr_at_k,
    needs_judge,
    page_recall_at_k,
    retrieval_metrics,
    score_question,
)
from kaprag.eval.pipeline import PipelineOutput, RetrievedItem
from kaprag.eval.schema import load_questions
from kaprag.eval.summary import mean_metric, percentile, summarize, summarize_group

FIXTURE = Path(__file__).parent / "fixtures" / "eval" / "questions.jsonl"
Q = {q.id: q for q in load_questions(FIXTURE)}


def item(doc, start, end=None, rank=0):
    return RetrievedItem(
        chunk_id=f"c{rank}", doc=doc, page_start=start, page_end=end or start, score=1.0
    )


def filler(n):
    return [item("akbank-2024", 900 + i, rank=i) for i in range(n)]


# --- retrieval -------------------------------------------------------------

GOLD = [("isbank-2024", 120)]


def test_recall_at_k_respects_cutoff():
    retrieved = filler(5) + [item("isbank-2024", 120)] + filler(4)
    assert page_recall_at_k(GOLD, retrieved, 5) == 0.0
    assert page_recall_at_k(GOLD, retrieved, 10) == 1.0
    assert mrr_at_k(GOLD, retrieved, 10) == pytest.approx(1 / 6)


def test_recall_hit_at_rank_one():
    retrieved = [item("isbank-2024", 120)] + filler(9)
    assert page_recall_at_k(GOLD, retrieved, 5) == 1.0
    assert mrr_at_k(GOLD, retrieved, 10) == 1.0


def test_hit_uses_page_range_and_doc():
    assert page_recall_at_k(GOLD, [item("isbank-2024", 118, 121)], 5) == 1.0
    assert page_recall_at_k(GOLD, [item("isbank-2024", 121, 125)], 5) == 0.0
    assert page_recall_at_k(GOLD, [item("isbank-2023", 120)], 5) == 0.0


def test_beyond_k_is_a_miss():
    retrieved = filler(10) + [item("isbank-2024", 120)]
    assert page_recall_at_k(GOLD, retrieved, 10) == 0.0
    assert mrr_at_k(GOLD, retrieved, 10) == 0.0


def test_empty_retrieval_and_no_gold():
    assert page_recall_at_k(GOLD, [], 5) == 0.0
    with pytest.raises(ValueError):
        page_recall_at_k([], filler(3), 5)


def test_all_evidence_recall():
    gold = [("garanti-2025", 50), ("akbank-2025", 60)]
    one = [item("garanti-2025", 50)] + filler(9)
    both = [item("garanti-2025", 50)] + filler(8) + [item("akbank-2025", 60)]
    assert all_evidence_recall_at_k(gold, one, 10) == 0.0
    assert all_evidence_recall_at_k(gold, both, 10) == 1.0
    assert page_recall_at_k(gold, one, 10) == 1.0


def test_retrieval_metrics_applicability():
    assert retrieval_metrics([], filler(3)) == {
        "page_recall@5": None,
        "page_recall@10": None,
        "mrr@10": None,
        "all_evidence_recall@10": None,
    }
    single = retrieval_metrics(GOLD, [item("isbank-2024", 120)])
    assert single["all_evidence_recall@10"] is None
    assert single["page_recall@5"] == 1.0


# --- citations -------------------------------------------------------------


def test_citation_precision_recall():
    gold = [("isbank-2024", 120), ("isbank-2024", 121)]
    cites = [Citation(doc="isbank-2024", page=120), Citation(doc="isbank-2024", page=5)]
    assert citation_scores(gold, cites) == (0.5, 0.5)
    dup = [Citation(doc="isbank-2024", page=120)] * 2
    assert citation_scores(gold, dup) == (1.0, 0.5)
    assert citation_scores(gold, []) == (None, 0.0)


# --- behaviour and applicability -------------------------------------------


def ans(**kw):
    return SystemAnswer(**({"answer": "x"} | kw))


def test_behavior_rules():
    assert behavior_correct(Q["f007"], ans(abstained=True))
    assert not behavior_correct(Q["f007"], ans())
    assert behavior_correct(Q["f008"], ans(clarification="Hangi banka?"))
    assert not behavior_correct(Q["f008"], ans(clarification="  "))
    assert not behavior_correct(Q["f008"], ans(clarification="Hangi?", abstained=True))
    assert behavior_correct(Q["f002"], ans())
    assert not behavior_correct(Q["f002"], ans(abstained=True))
    assert not behavior_correct(Q["f002"], ans(clarification="Hangi yıl?"))
    assert behavior_correct(Q["f009"], ans(basis="konsolide"))
    assert not behavior_correct(Q["f009"], ans(basis="solo"))
    assert not behavior_correct(Q["f009"], ans(basis=None))


@pytest.mark.parametrize(
    ("qid", "metric"),
    [
        ("f001", "text"),  # olgusal is judged even with a value
        ("f002", "numeric"),
        ("f003", "numeric"),
        ("f004", "numeric"),
        ("f005", "text"),  # non-numeric karsilastirma
        ("f006", "key_points"),
        ("f007", None),
        ("f008", None),
        ("f009", "numeric"),  # belirsiz with a value
    ],
)
def test_content_metric(qid, metric):
    assert content_metric(Q[qid]) == metric


def out(answer, retrieved=()):
    return PipelineOutput(answer=answer, retrieved=list(retrieved))


def test_needs_judge():
    assert needs_judge(Q["f001"], out(ans())) == "text"
    assert needs_judge(Q["f006"], out(ans())) == "key_points"
    assert needs_judge(Q["f001"], out(ans(abstained=True))) is None
    assert needs_judge(Q["f001"], None) is None
    assert needs_judge(Q["f002"], out(ans())) is None


# --- score_question --------------------------------------------------------


def test_numeric_question_correct_with_scale_word():
    output = out(
        ans(value=75.3, unit="milyar TL", citations=[Citation(doc="isbank-2024", page=120)]),
        [item("isbank-2024", 120)],
    )
    m = score_question(Q["f002"], output)
    assert m["numeric_correct"] is True
    assert m["numeric_value"] == 75_300_000_000
    assert m["answer_score"] == 1.0
    assert m["citation_precision"] == 1.0
    assert m["citation_recall"] == 0.5
    assert m["page_recall@5"] == 1.0
    assert m["all_evidence_recall@10"] == 0.0
    assert m["false_abstain"] is False
    assert m["correct_abstain"] is None


def test_failed_question_scores_as_wrong():
    for qid in ("f001", "f002", "f006", "f007", "f008", "f009"):
        m = score_question(Q[qid], None)
        assert m["answer_score"] == 0.0, qid
        assert m["behavior_correct"] is False
    m = score_question(Q["f002"], None)
    assert (m["numeric_correct"], m["numeric_reason"]) == (False, "pipeline error")
    assert m["page_recall@10"] == 0.0
    assert m["citation_recall"] == 0.0
    assert score_question(Q["f001"], None)["text_correct"] is False
    assert score_question(Q["f006"], None)["key_point_coverage"] == 0.0
    assert score_question(Q["f007"], None)["correct_abstain"] is False
    assert score_question(Q["f001"], None)["false_abstain"] is None


def test_false_abstain_on_answer_question():
    m = score_question(Q["f001"], out(ans(abstained=True)))
    assert m["false_abstain"] is True
    assert m["text_correct"] is False
    assert m["answer_score"] == 0.0


def test_correct_abstain():
    m = score_question(Q["f007"], out(ans(abstained=True)))
    assert m["correct_abstain"] is True
    assert m["answer_score"] == 1.0
    assert m["page_recall@5"] is None
    assert m["citation_precision"] is None


def test_clarify():
    assert score_question(Q["f008"], out(ans(clarification="Hangi banka?")))["answer_score"] == 1
    m = score_question(Q["f008"], out(ans()))
    assert (m["clarify_correct"], m["answer_score"]) == (False, 0.0)


def test_answer_with_assumption_needs_basis_and_value():
    good = ans(value=42.1, unit="milyar TL", basis="konsolide")
    assert score_question(Q["f009"], out(good))["answer_score"] == 1.0
    wrong_basis = ans(value=42.1, unit="milyar TL", basis="solo")
    m = score_question(Q["f009"], out(wrong_basis))
    assert (m["assumption_correct"], m["numeric_correct"], m["answer_score"]) == (
        False,
        True,
        0.0,
    )


def test_judge_results_flow_into_score():
    assert score_question(Q["f001"], out(ans()), text_correct=True)["answer_score"] == 1.0
    assert score_question(Q["f001"], out(ans()), text_correct=False)["answer_score"] == 0.0
    missing = score_question(Q["f001"], out(ans()), text_correct=None)
    assert missing["text_correct"] is None and missing["answer_score"] is None
    m = score_question(Q["f006"], out(ans()), key_point_coverage=0.5)
    assert (m["key_point_coverage"], m["answer_score"]) == (0.5, 0.5)


# --- summary ---------------------------------------------------------------


def test_mean_metric_and_percentile():
    assert mean_metric([True, False, None, 1.0]) == {"value": 2 / 3, "n": 3}
    assert mean_metric([None]) == {"value": None, "n": 0}
    assert percentile([], 50) is None
    assert percentile([10.0], 95) == 10.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.5
    assert percentile([float(i) for i in range(1, 101)], 95) == pytest.approx(95.05)


def row(qid, *, status="ok", latency=100.0, eligible=True, cost=0.01, judge_cost=0.0):
    q = Q[qid]
    output = None if status == "error" else out(ans(abstained=q.type == "cevapsiz"))
    return {
        "id": qid,
        "type": q.type,
        "verified": q.verified,
        "status": status,
        "metrics": score_question(q, output, text_correct=True),
        "latency_ms": latency,
        "latency_eligible": eligible and status == "ok",
        "cost_usd": cost,
        "input_tokens": 10,
        "output_tokens": 5,
        "llm_calls": 1,
        "cached_llm_calls": 0 if eligible else 1,
        "judge_cost_usd": judge_cost,
        "judge_error": None,
    }


def test_summarize_group_operational():
    rows = [
        row("f001", latency=100.0),
        row("f007", latency=300.0),
        row("f012", latency=5.0, eligible=False),
        row("f005", status="error", latency=1.0),
    ]
    g = summarize_group(rows)
    assert (g["n"], g["n_ok"], g["n_error"]) == (4, 3, 1)
    op = g["operational"]
    assert op["latency_n"] == 2
    assert op["latency_ms_p50"] == 200.0
    assert op["latency_excluded_cached"] == 1
    assert op["latency_excluded_error"] == 1
    assert op["cost_usd_total"] == pytest.approx(0.04)
    assert op["cost_usd_per_question"] == pytest.approx(0.01)
    assert g["metrics"]["correct_abstain_rate"] == {"value": 1.0, "n": 2}
    # f005 failed: not an abstention, so not in the false-abstain denominator
    assert g["metrics"]["false_abstain_rate"] == {"value": 0.0, "n": 1}
    assert g["metrics"]["answer_accuracy"] == {"value": 0.75, "n": 4}


def test_summarize_verified_empty_gives_nulls():
    rows = [row("f003"), row("f008")]  # neither is verified
    s = summarize(rows)
    assert s["verified"]["n"] == 0
    assert s["verified"]["metrics"]["answer_accuracy"] == {"value": None, "n": 0}
    assert s["verified"]["operational"]["cost_usd_total"] is None
    assert s["verified"]["operational"]["latency_ms_p50"] is None
    assert set(s["by_type"]) == {
        "olgusal", "tablo", "hesaplama", "karsilastirma", "sentez", "cevapsiz", "belirsiz"
    }  # fmt: skip
    assert s["by_type"]["olgusal"]["metrics"]["answer_accuracy"]["value"] is None
    assert s["by_type"]["belirsiz"]["n"] == 1
    json.dumps(s)  # serialisable
