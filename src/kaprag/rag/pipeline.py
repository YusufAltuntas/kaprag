"""`RagPipeline`: retrieve -> numbered context -> one generation call -> section 5 answer.

Registered in the harness as `rag`; a config selects it with `factory: rag` and its
params (see `kaprag.rag.config`). Per question:

1. Retrieve the top `retrieve_k` chunks (in a worker thread; the retriever serialises
   its own GPU/DB work). This ranked list is what the harness scores for retrieval.
   When the retrieval stages record per-item ranks (hybrid arms, reranking), they go
   to `debug["retrieval_trace"]`, one entry per returned chunk, best first.
2. Put the first `context_k` of them into the prompt as sources `[1]..[context_k]`.
3. One schema-validated LLM call tagged `{**tags, "step": "generate"}`. A failure
   (`LLMCallError`) propagates, so the runner records the question as an error.
4. Map citations and normalise the value in code (`kaprag.rag.postprocess`).
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from kaprag.eval.pipeline import PipelineContext, PipelineOutput, register_pipeline
from kaprag.llm import LLMClient
from kaprag.rag.config import RagConfig, parse_rag_config
from kaprag.rag.context import format_context
from kaprag.rag.generate import generate
from kaprag.rag.postprocess import build_answer
from kaprag.rag.retriever import Retriever, open_retriever


class RagPipeline:
    def __init__(
        self, config: RagConfig, retriever: Retriever, llm: Callable[[], LLMClient]
    ) -> None:
        self.config = config
        self.retriever = retriever
        self.llm = llm

    async def run(self, question: str, *, tags: dict[str, Any]) -> PipelineOutput:
        ranked = await asyncio.to_thread(self.retriever.retrieve, question, self.config.retrieve_k)
        retrieved = [r.item for r in ranked]
        sources = retrieved[: self.config.context_k]
        generated = await generate(
            self.llm(),
            question,
            format_context(sources),
            model=self.config.generation.model,
            tags=tags,
        )
        answer, debug = build_answer(generated, sources)
        retrieval_debug = {}
        if any(r.trace for r in ranked):
            retrieval_debug["retrieval_trace"] = [
                {"chunk_id": r.item.chunk_id, **r.trace} for r in ranked
            ]
        return PipelineOutput(
            answer=answer,
            retrieved=retrieved,
            debug={
                "context_chunk_ids": [s.chunk_id for s in sources],
                **retrieval_debug,
                **debug,
            },
        )

    async def aclose(self) -> None:
        await asyncio.to_thread(self.retriever.close)


@register_pipeline("rag")
def build(ctx: PipelineContext) -> RagPipeline:
    """Harness factory. Loads the embedder and connects to the database once per run."""
    config = parse_rag_config(ctx.params)
    return RagPipeline(config, open_retriever(config), ctx.llm)
