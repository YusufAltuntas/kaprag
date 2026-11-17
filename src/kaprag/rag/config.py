"""RAG pipeline params: the `pipeline.params` block of `configs/<name>.yaml`.

    index: fixed              # chunk table: fixed | structure
    retrieval:
      mode: dense             # see kaprag.rag.retriever.RETRIEVAL_MODES
      # hybrid only:
      candidates_per_arm: 40  # top-n of each arm (dense, FTS) fed to the fusion
      rrf_k: 60               # reciprocal rank fusion constant (not tuned)
    retrieve_k: 10            # ranked items returned to the harness (recall@10, mrr@10)
    context_k: 5              # top items put into the prompt (<= retrieve_k)
    generation:
      model: claude-haiku-5-5
    embedding:
      model: BAAI/bge-m3
      device: cuda

Unknown keys and unknown values fail loudly; hybrid-only keys under another mode too.
A new retrieval mode is a new `mode` value registered in `kaprag.rag.retriever`; a
reranker or a metadata filter is a new optional block here plus a retriever stage, not
a change to generation. Each block is independent of the others, so an ablation
config changes one block of the previous one (e.g. `index: structure` under any mode).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from kaprag.rag.retriever import RETRIEVAL_MODES


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


_HYBRID_KEYS = frozenset({"candidates_per_arm", "rrf_k"})


class RetrievalConfig(_Strict):
    mode: str = "dense"
    candidates_per_arm: int = Field(default=40, ge=1)
    rrf_k: int = Field(default=60, ge=0)

    @model_validator(mode="after")
    def _check(self) -> RetrievalConfig:
        stray = sorted(_HYBRID_KEYS & self.model_fields_set)
        if self.mode != "hybrid" and stray:
            raise ValueError(f"{stray} only apply to retrieval mode 'hybrid'")
        return self


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
