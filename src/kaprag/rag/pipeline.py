"""`RagPipeline`: retrieve -> numbered context -> one generation call -> section 5 answer.

Registered in the harness as `rag`; a config selects it with `factory: rag` and its
params (see `kaprag.rag.config`). Per question:

0. With a `query_filter` block (enabled): one parse call tagged
   `{**tags, "step": "parse_query"}` turns the question's banks/years into a SQL filter
   for retrieval (`kaprag.rag.query_filter`); a failed parse means no filter, never an
   error. `debug["query_filter"]` records the parse, the applied filter, any failure
   and the number of chunks retrieved under it (`retrieved`).
0b. Access (simulated roles, `kaprag.access`): with an `access` block, or when the
   caller passes `run(..., access_role=...)` (which overrides the config, e.g. per
   request in a demo), `access_roles && <effective roles>` is ANDed into the filter
   (`kaprag.access.filter`), so both retrieval arms exclude restricted chunks before
   ranking. `debug["access"]` records the role, its effective roles and its source.
   The role never comes from the question. No block and no caller role: no condition.
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

from kaprag.access.filter import access_filter
from kaprag.access.roles import effective_roles
from kaprag.eval.pipeline import PipelineContext, PipelineOutput, register_pipeline
from kaprag.llm import LLMClient
from kaprag.rag.config import RagConfig, parse_rag_config
from kaprag.rag.context import format_context
from kaprag.rag.generate import generate
from kaprag.rag.postprocess import build_answer
from kaprag.rag.query_filter import parse_query_filter
from kaprag.rag.retriever import Retriever, open_retriever


class RagPipeline:
    def __init__(
        self, config: RagConfig, retriever: Retriever, llm: Callable[[], LLMClient]
    ) -> None:
        self.config = config
        self.retriever = retriever
        self.llm = llm

    async def run(
        self, question: str, *, tags: dict[str, Any], access_role: str | None = None
    ) -> PipelineOutput:
        role = self.request_role(access_role)
        where, filter_debug = None, None
        qf = self.config.query_filter
        if qf is not None and qf.enabled:
            outcome = await parse_query_filter(self.llm(), question, model=qf.model, tags=tags)
            where, filter_debug = outcome.where, outcome.debug
        if role is not None:
            where = access_filter(role, where)
        ranked = await asyncio.to_thread(
            self.retriever.retrieve, question, self.config.retrieve_k, where=where
        )
        if filter_debug is not None:
            filter_debug = {**filter_debug, "retrieved": len(ranked)}
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
        retrieval_debug: dict[str, Any] = {}
        if role is not None:
            retrieval_debug["access"] = {
                "role": role,
                "effective_roles": list(effective_roles(role)),
                "source": "caller" if access_role is not None else "config",
            }
        if filter_debug is not None:
            retrieval_debug["query_filter"] = filter_debug
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

    def request_role(self, access_role: str | None) -> str | None:
        """The access role of a request: the caller's `access_role` when given, else the
        config's `access.role`; None = no access condition. Never derived from the
        question."""
        if access_role is not None:
            effective_roles(access_role)  # refuse an unknown role before any work
            return access_role
        return self.config.access.role if self.config.access is not None else None

    async def aclose(self) -> None:
        await asyncio.to_thread(self.retriever.close)


@register_pipeline("rag")
def build(ctx: PipelineContext) -> RagPipeline:
    """Harness factory. Loads the embedder and connects to the database once per run."""
    config = parse_rag_config(ctx.params)
    return RagPipeline(config, open_retriever(config), ctx.llm)
