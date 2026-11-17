"""Per-question metrics (eval/README.md §6). Pure functions; the judge runs elsewhere.

Which metrics apply to a question:

- Retrieval (`page_recall@5`, `page_recall@10`, `mrr@10`): questions with at least one
  gold evidence page; `all_evidence_recall@10` only with two or more gold pages. A
  retrieved item hits a gold page when the doc matches and the page lies within
  `page_start..page_end`. Otherwise these are None (excluded; denominators reported).
- Content (`content_metric`), for `answer` and `answer_with_assumption` questions:
  `numeric` when `expected.value` is set and the type is `tablo`, `hesaplama`,
  `karsilastirma` or `belirsiz`; `key_points` (judge) for `sentez`; `text` (judge)
  otherwise (`olgusal`, non-numeric `karsilastirma`, and value-less `tablo` /
  `hesaplama` / `belirsiz`). `olgusal` is always judged, as the spec says.
- Citations: questions with gold pages whose expected behaviour is to answer.
  Precision is None when the answer cites nothing (recall is then 0).
- Behaviour (deterministic, from the structured answer):
  `abstain` -> `abstained`; `answer` -> not abstained and no clarification;
  `clarify` -> not abstained and a non-empty `clarification`;
  `answer_with_assumption` -> not abstained, no clarification, and `basis` equal to
  the gold `basis`. `correct_abstain` is set on `abstain` questions, `false_abstain`
  on `answer` questions; both are None elsewhere.

A question whose pipeline failed (`output is None`) scores as an empty answer with no
retrieved items: wrong in every accuracy metric (including `correct_abstain`), 0 in
retrieval and citation recall. It is not counted in `false_abstain` (a failure is not
an abstention; errors are reported separately).

Headline `answer_score` (one number per question, in [0, 1]):
- `abstain`: 1 if the system abstained, else 0.
- `clarify`: 1 if it asked for clarification (behaviour check), else 0.
- `answer` / `answer_with_assumption`: 0 if the behaviour check fails; otherwise the
  content metric: numeric correct (1/0), judge text verdict (1/0), or key-point
  coverage (fraction) for `sentez`. None only when the judge verdict is missing
  (judge error) for an otherwise valid answer; such questions are excluded from the
  mean and the count is reported.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from kaprag.eval.answer import Citation, SystemAnswer
from kaprag.eval.numeric import compare
from kaprag.eval.pipeline import PipelineOutput, RetrievedItem
from kaprag.eval.schema import Question

GoldPage = tuple[str, int]

NUMERIC_TYPES = frozenset({"tablo", "hesaplama", "karsilastirma", "belirsiz"})
ANSWERING = frozenset({"answer", "answer_with_assumption"})


# --- retrieval -------------------------------------------------------------


def _first_hit_rank(gold: Sequence[GoldPage], retrieved: Sequence[RetrievedItem], k: int) -> int:
    """1-based rank of the first item in the top k hitting any gold page; 0 if none."""
    for rank, item in enumerate(retrieved[:k], start=1):
        if any(item.covers(doc, page) for doc, page in gold):
            return rank
    return 0


def page_recall_at_k(gold: Sequence[GoldPage], retrieved: Sequence[RetrievedItem], k: int) -> float:
    """1.0 if any of the top k items hits any gold page, else 0.0."""
    if not gold:
        raise ValueError("page_recall@k needs at least one gold page")
    return 1.0 if _first_hit_rank(gold, retrieved, k) else 0.0


def all_evidence_recall_at_k(
    gold: Sequence[GoldPage], retrieved: Sequence[RetrievedItem], k: int
) -> float:
    """1.0 if every gold page is hit by some item in the top k, else 0.0."""
    if not gold:
        raise ValueError("all_evidence_recall@k needs at least one gold page")
    top = retrieved[:k]
    return 1.0 if all(any(i.covers(d, p) for i in top) for d, p in gold) else 0.0


def mrr_at_k(gold: Sequence[GoldPage], retrieved: Sequence[RetrievedItem], k: int) -> float:
    """Reciprocal rank of the first top-k hit (0.0 when there is none)."""
    if not gold:
        raise ValueError("mrr@k needs at least one gold page")
    rank = _first_hit_rank(gold, retrieved, k)
    return 1.0 / rank if rank else 0.0


def retrieval_metrics(
    gold: Sequence[GoldPage], retrieved: Sequence[RetrievedItem]
) -> dict[str, float | None]:
    if not gold:
        return {
            "page_recall@5": None,
            "page_recall@10": None,
            "mrr@10": None,
            "all_evidence_recall@10": None,
        }
    return {
        "page_recall@5": page_recall_at_k(gold, retrieved, 5),
        "page_recall@10": page_recall_at_k(gold, retrieved, 10),
        "mrr@10": mrr_at_k(gold, retrieved, 10),
        "all_evidence_recall@10": (
            all_evidence_recall_at_k(gold, retrieved, 10) if len(gold) >= 2 else None
        ),
    }


# --- citations -------------------------------------------------------------


def citation_scores(
    gold: Sequence[GoldPage], citations: Sequence[Citation]
) -> tuple[float | None, float]:
    """(precision, recall) of distinct cited pages against gold pages.

    Precision is None when nothing is cited.
    """
    if not gold:
        raise ValueError("citation scores need at least one gold page")
    gold_set = set(gold)
    cited = {(c.doc, c.page) for c in citations}
    hits = len(cited & gold_set)
    precision = hits / len(cited) if cited else None
    return precision, hits / len(gold_set)


# --- behaviour and content ---------------------------------------------------


def behavior_correct(question: Question, answer: SystemAnswer) -> bool:
    asked = bool(answer.clarification and answer.clarification.strip())
    expected = question.expected_behavior
    if expected == "abstain":
        return answer.abstained
    if expected == "clarify":
        return not answer.abstained and asked
    if expected == "answer":
        return not answer.abstained and not asked
    # answer_with_assumption
    return not answer.abstained and not asked and answer.basis == question.basis


def content_metric(question: Question) -> str | None:
    """`numeric`, `text`, `key_points`, or None for abstain/clarify questions."""
    if question.expected_behavior not in ANSWERING:
        return None
    if question.type == "sentez":
        return "key_points"
    if question.expected.value is not None and question.type in NUMERIC_TYPES:
        return "numeric"
    return "text"


def needs_judge(question: Question, output: PipelineOutput | None) -> str | None:
    """The judge task (`text` / `key_points`) to run for this question, if any.

    No judge call for failed or abstained answers: they score 0 without one.
    """
    metric = content_metric(question)
    if metric not in ("text", "key_points") or output is None or output.answer.abstained:
        return None
    return metric


EMPTY_ANSWER = SystemAnswer(answer="")


def score_question(
    question: Question,
    output: PipelineOutput | None,
    *,
    text_correct: bool | None = None,
    key_point_coverage: float | None = None,
) -> dict[str, Any]:
    """All per-question metrics. Judge results are passed in (None = not available)."""
    failed = output is None
    answer = EMPTY_ANSWER if failed else output.answer
    retrieved = [] if failed else output.retrieved
    gold = question.gold_pages
    expected = question.expected_behavior

    metrics: dict[str, Any] = retrieval_metrics(gold, retrieved)

    behaves = False if failed else behavior_correct(question, answer)
    metrics["behavior_correct"] = behaves
    metrics["correct_abstain"] = (
        (not failed and answer.abstained) if expected == "abstain" else None
    )
    metrics["false_abstain"] = answer.abstained if expected == "answer" and not failed else None
    metrics["clarify_correct"] = behaves if expected == "clarify" else None
    metrics["assumption_correct"] = behaves if expected == "answer_with_assumption" else None

    if gold and expected in ANSWERING:
        precision, recall = citation_scores(gold, answer.citations)
    else:
        precision = recall = None
    metrics["citation_precision"] = precision
    metrics["citation_recall"] = recall

    metric = content_metric(question)
    metrics["content_metric"] = metric
    metrics["numeric_correct"] = None
    metrics["numeric_reason"] = None
    metrics["numeric_value"] = None
    metrics["text_correct"] = None
    metrics["key_point_coverage"] = None
    content: float | None = None
    skip_judge = failed or answer.abstained
    if metric == "numeric" and failed:
        metrics["numeric_correct"] = False
        metrics["numeric_reason"] = "pipeline error"
        content = 0.0
    elif metric == "numeric":
        exp = question.expected
        result = compare(
            answer.value,
            answer.unit,
            exp.value,
            exp.unit,
            exp.tolerance.type,
            exp.tolerance.amount,
        )
        metrics["numeric_correct"] = result.correct
        metrics["numeric_reason"] = result.reason
        metrics["numeric_value"] = result.value
        content = 1.0 if result.correct else 0.0
    elif metric == "text":
        verdict = False if skip_judge else text_correct
        metrics["text_correct"] = verdict
        content = None if verdict is None else float(verdict)
    elif metric == "key_points":
        coverage = 0.0 if skip_judge else key_point_coverage
        metrics["key_point_coverage"] = coverage
        content = coverage

    if expected in ANSWERING:
        metrics["answer_score"] = content if behaves else 0.0
    else:
        metrics["answer_score"] = 1.0 if behaves else 0.0
    return metrics
