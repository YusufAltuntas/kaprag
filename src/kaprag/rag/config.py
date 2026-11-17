"""RAG pipeline params: the `pipeline.params` block of `configs/<name>.yaml`.

    index: fixed              # chunk table: fixed | structure
    retrieval:
      mode: dense             # see kaprag.rag.retriever.RETRIEVAL_MODES
    retrieve_k: 10            # ranked items returned to the harness (recall@10, mrr@10)
    context_k: 5              # top items put into the prompt (<= retrieve_k)
    generation:
      model: claude-haiku-5-5
    embedding:
      model: BAAI/bge-m3
      device: cuda

Unknown keys and unknown values fail loudly. A new retrieval mode (e.g. `hybrid`) is
a new `mode` value registered in `kaprag.rag.retriever`; a reranker or a metadata
filter is a new optional block here plus a retriever stage, not a change to generation.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from kaprag.rag.retriever import RETRIEVAL_MODES


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RetrievalConfig(_Strict):
    mode: str = "dense"


class GenerationConfig(_Strict):
    model: str = "claude-haiku-5-5"


class EmbeddingConfig(_Strict):
    model: str = "BAAI/bge-m3"
    device: str = "cuda"


class RagConfig(_Strict):
    index: Literal["fixed", "structure"]
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    retrieve_k: int = Field(default=10, ge=1)
    context_k: int = Field(default=5, ge=1)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)

    @model_validator(mode="after")
    def _check(self) -> RagConfig:
        if self.retrieval.mode not in RETRIEVAL_MODES:
            raise ValueError(
                f"unknown retrieval mode {self.retrieval.mode!r}; "
                f"expected one of {sorted(RETRIEVAL_MODES)}"
            )
        if self.context_k > self.retrieve_k:
            raise ValueError(f"context_k ({self.context_k}) > retrieve_k ({self.retrieve_k})")
        return self


class RagConfigError(ValueError):
    pass


def parse_rag_config(params: dict[str, Any]) -> RagConfig:
    try:
        return RagConfig.model_validate(params)
    except ValidationError as exc:
        raise RagConfigError(f"invalid RAG pipeline params: {exc}") from None
