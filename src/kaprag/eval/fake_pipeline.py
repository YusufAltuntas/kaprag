"""FAKE "oracle" pipeline for testing the eval harness end to end.

It is not a RAG system: it reads the gold record of the question (through
`PipelineContext.gold`) and echoes it back, corrupting a deterministic, seeded fraction
of answers and retrievals so that the metrics are non-trivial. No retrieval, no LLM,
no database, no GPU. Its numbers say nothing about any real pipeline.

Params: `error_rate` (share of corrupted answers, and independently of corrupted
retrievals), `fail_rate` (share of questions that raise, to exercise error handling),
`seed`.
"""

from __future__ import annotations

import random
from typing import Any

from kaprag.eval.answer import Citation, SystemAnswer
from kaprag.eval.pipeline import (
    PipelineContext,
    PipelineOutput,
    RetrievedItem,
    register_pipeline,
)
from kaprag.eval.schema import Question

TOP_K = 10
FILLER_DOC = "akbank-2024"


class FakePipelineFailure(RuntimeError):
    """Raised on purpose for a `fail_rate` share of questions."""


class FakeOraclePipeline:
    def __init__(
        self, gold: dict[str, Question], *, error_rate: float, fail_rate: float, seed: int
    ) -> None:
        self.gold = gold
        self.error_rate = error_rate
        self.fail_rate = fail_rate
        self.seed = seed

    async def run(self, question: str, *, tags: dict[str, Any]) -> PipelineOutput:
        gold = self.gold[tags["qid"]]
        rng = random.Random(f"{self.seed}:{gold.id}")
        if rng.random() < self.fail_rate:
            raise FakePipelineFailure(f"fake failure for {gold.id}")
        wrong_answer = rng.random() < self.error_rate
        wrong_retrieval = rng.random() < self.error_rate
        return PipelineOutput(
            answer=_answer(gold, wrong_answer),
            retrieved=_retrieved(gold, wrong_retrieval, rng),
        )


def _answer(gold: Question, wrong: bool) -> SystemAnswer:
    behavior = gold.expected_behavior
    if behavior == "abstain":
        if wrong:
            return SystemAnswer(answer="Uydurma bir cevap.", abstained=False)
        return SystemAnswer(answer="Bu bilgi raporlarda yok.", abstained=True)
    if behavior == "clarify":
        if wrong:
            return SystemAnswer(answer=gold.expected.answer)
        return SystemAnswer(answer="", clarification="Hangi banka ve yılı kastediyorsunuz?")

    text = " ".join([gold.expected.answer, *gold.key_points])
    value, unit = gold.expected.value, gold.expected.unit
    if value is not None and unit == "TL" and abs(value) >= 1e9:
        value, unit = value / 1e9, "milyar TL"  # exercise unit normalisation
    pages = gold.gold_pages
    if wrong:
        text = "Yanlış bir cevap."
        value = None if value is None else value * 1.5
        pages = [(doc, page + 7) for doc, page in pages]
    basis = gold.basis
    if wrong and basis is not None:
        basis = "solo" if basis == "konsolide" else "konsolide"
    return SystemAnswer(
        answer=text,
        value=value,
        unit=unit,
        basis=basis,
        citations=[Citation(doc=doc, page=page) for doc, page in pages],
    )


def _retrieved(gold: Question, wrong: bool, rng: random.Random) -> list[RetrievedItem]:
    pages = [] if wrong else list(gold.gold_pages)
    first = rng.randint(0, 2)
    slots: list[tuple[str, int] | None] = [None] * TOP_K
    for i, page in enumerate(pages[: TOP_K - first]):
        slots[first + i] = page
    items = []
    for rank, slot in enumerate(slots, start=1):
        doc, page = slot if slot is not None else (FILLER_DOC, 900 + rank)
        text = f"fake chunk {doc} p{page}"
        items.append(
            RetrievedItem(
                chunk_id=f"fake-{doc}-p{page}-r{rank}",
                doc=doc,
                page_start=page,
                page_end=page,
                score=round(1.0 - 0.05 * rank, 4),
                text=text,
            )
        )
    return items


@register_pipeline("fake_oracle")
def build_fake_oracle(ctx: PipelineContext) -> FakeOraclePipeline:
    params = ctx.params
    return FakeOraclePipeline(
        ctx.gold,
        error_rate=float(params.get("error_rate", 0.25)),
        fail_rate=float(params.get("fail_rate", 0.0)),
        seed=int(params.get("seed", 42)),
    )
