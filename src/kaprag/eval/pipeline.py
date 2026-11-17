"""The interface between the eval runner and a RAG pipeline, plus the pipeline registry.

A pipeline is built by a factory from a `PipelineContext` and answers one question at
a time:

    @register_pipeline("baseline")
    def build(ctx: PipelineContext) -> Pipeline: ...

    output = await pipeline.run(question_text, tags=tags)

`tags` is opaque to the pipeline but MUST be merged into the `tags=` of every LLM
call it makes (`client.complete(..., tags={**tags, "step": "generate"})`): the runner
attributes LLM calls (cost, cache hits) to a question through `tags["qid"]`.

`PipelineOutput.retrieved` is the ranked list actually considered, best first; return
at least the top 10 even when generating from fewer, since retrieval metrics are
computed @5 and @10 over this list.

A config names its factory either by registry key (`fake_oracle`) or by dotted path
(`kaprag.rag.baseline:build`); the dotted form needs no edit here.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field, model_validator

from kaprag.eval.answer import SystemAnswer
from kaprag.eval.schema import Question
from kaprag.llm import LLMClient


class RetrievedItem(BaseModel):
    """One retrieved chunk. Pages are 1-based PDF page indices, inclusive."""

    chunk_id: str
    doc: str
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    score: float
    text: str | None = None
    text_chars: int | None = None  # filled from `text` when omitted

    @model_validator(mode="after")
    def _check(self) -> RetrievedItem:
        if self.page_end < self.page_start:
            raise ValueError("page_end must be >= page_start")
        if self.text_chars is None and self.text is not None:
            self.text_chars = len(self.text)
        return self

    def covers(self, doc: str, page: int) -> bool:
        return self.doc == doc and self.page_start <= page <= self.page_end


class PipelineOutput(BaseModel):
    answer: SystemAnswer
    retrieved: list[RetrievedItem]
    # Free-form, JSON-serialisable diagnostics (e.g. why a value was dropped); written
    # to the run's traces.jsonl, never scored.
    debug: dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class Pipeline(Protocol):
    async def run(self, question: str, *, tags: dict[str, Any]) -> PipelineOutput: ...


@dataclass
class PipelineContext:
    """What a factory gets.

    `llm()` returns the run's shared LLM client (created on first use; its trace is the
    run's trace). `gold` maps question id to gold record and exists ONLY for fake /
    oracle pipelines that test the harness; a real pipeline must never read it.
    """

    params: dict[str, Any]
    llm: Callable[[], LLMClient]
    gold: dict[str, Question] = field(default_factory=dict)


PipelineFactory = Callable[[PipelineContext], Pipeline]

_REGISTRY: dict[str, PipelineFactory] = {}
# Modules whose import registers built-in pipelines.
_BUILTIN_MODULES = ("kaprag.eval.fake_pipeline", "kaprag.rag.pipeline", "kaprag.agent.pipeline")


def register_pipeline(name: str) -> Callable[[PipelineFactory], PipelineFactory]:
    """Decorator registering a factory under `name`."""

    def decorator(factory: PipelineFactory) -> PipelineFactory:
        if name in _REGISTRY and _REGISTRY[name] is not factory:
            raise ValueError(f"pipeline {name!r} is already registered")
        _REGISTRY[name] = factory
        return factory

    return decorator


def resolve_factory(ref: str) -> PipelineFactory:
    """Look up a registry key, or import `package.module:attribute`."""
    if ":" in ref:
        module_name, _, attr = ref.partition(":")
        factory = getattr(importlib.import_module(module_name), attr, None)
        if not callable(factory):
            raise ValueError(f"pipeline factory {ref!r} not found or not callable")
        return factory
    for module_name in _BUILTIN_MODULES:
        importlib.import_module(module_name)
    if ref not in _REGISTRY:
        raise ValueError(f"unknown pipeline {ref!r}; registered: {sorted(_REGISTRY)}")
    return _REGISTRY[ref]
