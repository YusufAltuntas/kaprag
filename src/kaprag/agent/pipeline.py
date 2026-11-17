"""`RouterPipeline`: the agent router (`kaprag.agent.graph`) as a harness pipeline.

Registered as `router`; `configs/router.yaml` selects it. Params:

    rag:                       # the simple path AND the look-ups' retrieval stack:
      index: fixed             # a full `kaprag.rag.config` block (e.g. rerank's params,
      ...                      # with its optional `access` block, which then applies
                               # to the simple path and to every look-up)
    classify:
      model: claude-haiku-5-5
    decompose:
      model: claude-sonnet-5-5
      max_lookups: 8           # 2..8 (`kaprag.agent.plan.MAX_LOOKUPS`)
    synthesize:
      model: claude-haiku-5-5

`run(question, tags=..., access_role=...)` behaves like `RagPipeline.run`: the
caller's role overrides the config's; an unknown role is refused before any work.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from kaprag.agent.graph import RouterModels, build_graph
from kaprag.agent.plan import MAX_LOOKUPS
from kaprag.eval.pipeline import PipelineContext, PipelineOutput, register_pipeline
from kaprag.llm import LLMClient
from kaprag.rag.config import RagConfig
from kaprag.rag.pipeline import RagPipeline
from kaprag.rag.retriever import Retriever, open_retriever

HAIKU = "claude-haiku-5-5"
SONNET = "claude-sonnet-5-5"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class StepModel(_Strict):
    model: str


class DecomposeConfig(_Strict):
    model: str = SONNET
    max_lookups: int = Field(default=MAX_LOOKUPS, ge=2, le=MAX_LOOKUPS)


class RouterConfig(_Strict):
    rag: RagConfig
    classify: StepModel = Field(default_factory=lambda: StepModel(model=HAIKU))
    decompose: DecomposeConfig = Field(default_factory=DecomposeConfig)
    synthesize: StepModel = Field(default_factory=lambda: StepModel(model=HAIKU))

    def models(self) -> RouterModels:
        return RouterModels(
            classify=self.classify.model,
            decompose=self.decompose.model,
            synthesize=self.synthesize.model,
            max_lookups=self.decompose.max_lookups,
        )


class RouterConfigError(ValueError):
    pass


def parse_router_config(params: dict[str, Any]) -> RouterConfig:
    try:
        return RouterConfig.model_validate(params)
    except ValidationError as exc:
        raise RouterConfigError(f"invalid router pipeline params: {exc}") from None


class RouterPipeline:
    def __init__(
        self, config: RouterConfig, retriever: Retriever, llm: Callable[[], LLMClient]
    ) -> None:
        self.config = config
        self.rag = RagPipeline(config.rag, retriever, llm)
        self.graph = build_graph(self.rag, config.models())

    async def run(
        self, question: str, *, tags: dict[str, Any], access_role: str | None = None
    ) -> PipelineOutput:
        role = self.rag.request_role(access_role)
        state = await self.graph.ainvoke(
            {
                "question": question,
                "tags": dict(tags),
                "access_role": access_role,
                "role": role,
                "lookups": [],
            }
        )
        return state["output"]

    async def aclose(self) -> None:
        await self.rag.aclose()


@register_pipeline("router")
def build(ctx: PipelineContext) -> RouterPipeline:
    """Harness factory. Loads the embedder (and reranker) and connects once per run."""
    config = parse_router_config(ctx.params)
    return RouterPipeline(config, open_retriever(config.rag), ctx.llm)
