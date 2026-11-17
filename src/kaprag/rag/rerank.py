"""Cross-encoder reranking: a stage after any first-stage retrieval mode.

The top `candidates` of the first-stage ranking are scored as (question, chunk text)
pairs by a cross-encoder and sorted by that score (ties keep the first-stage order);
the pipeline then takes `retrieve_k` / `context_k` from the reranked list as usual. An
item's score becomes the reranker score; its trace keeps the first-stage rank and
score (`first_stage_rank`, `first_stage_score`) next to the first stage's own fields.

The model (`BAAI/bge-reranker-v2-m3` by default; needs the `ml` group, imported
lazily) loads once per run in fp16 on CUDA, with no silent CPU fallback, and scores in
small batches with a capped pair length. The cap (1024 tokens) fits a fixed chunk plus
the question and special tokens without truncation: the longest pair measured (bge-m3
tokenizer, which the reranker shares; 30 dev questions x 40 candidates) was 547 tokens.
Like the embedder, one model is not safe for concurrent use, so scoring holds a lock;
the pipeline runs retrieval, reranking included, in a worker thread, off the event
loop.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from typing import Any, Protocol

from kaprag.rag.retriever import RankedChunk, Retriever
from kaprag.retrieval import SqlFilter

DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
DEFAULT_CANDIDATES = 40
DEFAULT_BATCH_SIZE = 8
DEFAULT_MAX_LENGTH = 1024


class Scorer(Protocol):
    def score(self, question: str, passages: Sequence[str]) -> list[float]: ...


class CrossEncoderScorer:
    def __init__(
        self,
        model_name: str = DEFAULT_RERANK_MODEL,
        *,
        device: str = "cuda",
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_length: int = DEFAULT_MAX_LENGTH,
    ) -> None:
        import torch
        from sentence_transformers import CrossEncoder

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available; pass device='cpu' explicitly to use CPU")
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.model: Any = CrossEncoder(
            model_name,
            device=device,
            max_length=max_length,
            model_kwargs={"torch_dtype": dtype},
        )
        self._lock = threading.Lock()

    def score(self, question: str, passages: Sequence[str]) -> list[float]:
        """Relevance of each passage to the question (higher is more relevant)."""
        if not passages:
            return []
        with self._lock:
            scores = self.model.predict(
                [(question, p) for p in passages],
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        return [float(s) for s in scores]


def rerank(
    question: str, ranked: Sequence[RankedChunk], scorer: Scorer, candidates: int
) -> list[RankedChunk]:
    """The top `candidates` of `ranked`, sorted by the scorer, best first."""
    pool = list(ranked[:candidates])
    if not pool:
        return []
    texts = []
    for r in pool:
        if r.item.text is None:
            raise ValueError(f"chunk {r.item.chunk_id} has no text to rerank")
        texts.append(r.item.text)
    scores = scorer.score(question, texts)
    order = sorted(range(len(pool)), key=lambda i: (-scores[i], i))
    return [
        RankedChunk(
            pool[i].item.model_copy(update={"score": scores[i]}),
            {
                **pool[i].trace,
                "first_stage_rank": i + 1,
                "first_stage_score": pool[i].item.score,
            },
        )
        for i in order
    ]


class RerankingRetriever:
    """A first-stage retriever followed by cross-encoder reranking of its top
    `candidates` (config `rerank.candidates` >= `retrieve_k`)."""

    def __init__(self, first_stage: Retriever, scorer: Scorer, candidates: int) -> None:
        self.first_stage = first_stage
        self.scorer = scorer
        self.candidates = candidates

    def retrieve(self, question: str, k: int, where: SqlFilter | None = None) -> list[RankedChunk]:
        first = self.first_stage.retrieve(question, self.candidates, where=where)
        return rerank(question, first, self.scorer, self.candidates)[:k]

    def close(self) -> None:
        self.first_stage.close()
