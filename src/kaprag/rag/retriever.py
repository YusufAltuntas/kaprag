"""Retrieval modes and their construction from the config.

A `Retriever` returns the top-k chunks for a question, best first, as `RankedChunk`s:
the harness's `RetrievedItem` (text included) plus a per-item `trace` of how the
stages ranked it (e.g. `{"dense_rank": 3, "fts_rank": None}`), which the pipeline
writes to its debug output. It is synchronous and may block (GPU embedding, a database
query); `kaprag.rag.pipeline` runs it in a worker thread.

`RETRIEVAL_MODES` maps a config `retrieval.mode` to an opener:

- `dense` (the baseline): pgvector cosine search; the trace is empty.
- `hybrid`: the dense arm and the Turkish FTS arm over the same chunk table, the top
  `candidates_per_arm` of each fused by reciprocal rank fusion (`rrf_k`). A chunk found
  by one arm only is kept; the item's score is the fused score.

`open_retriever` wraps any mode in the cross-encoder reranking stage
(`kaprag.rag.rerank`) when the config has a `rerank` block; a SQL filter would
parameterise a retriever here the same way. Generation never changes for them.

Concurrency: a retriever owns one embedder and one autocommit connection, neither
safe for concurrent use, so `retrieve` holds a lock: retrievals are serialised (each
is tens of milliseconds next to seconds of generation).
"""

from __future__ import annotations

import dataclasses
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from kaprag.eval.pipeline import RetrievedItem
from kaprag.retrieval import SearchHit, dense_search, fts_search, rrf_fuse

if TYPE_CHECKING:
    from psycopg import Connection

    from kaprag.rag.config import RagConfig


@dataclass(frozen=True)
class RankedChunk:
    item: RetrievedItem
    trace: dict[str, Any] = field(default_factory=dict)


class Retriever(Protocol):
    def retrieve(self, question: str, k: int) -> list[RankedChunk]: ...

    def close(self) -> None: ...


class _IndexRetriever:
    """Shared state of the retrievers over one chunk table."""

    def __init__(self, conn: Connection, variant: str, embedder: Any) -> None:
        self.conn = conn
        self.variant = variant
        self.embedder = embedder
        self._lock = threading.Lock()

    def close(self) -> None:
        with self._lock:
            self.conn.close()


class DenseRetriever(_IndexRetriever):
    """Dense (pgvector cosine) search over one chunk table."""

    def retrieve(self, question: str, k: int) -> list[RankedChunk]:
        with self._lock:
            vector = self.embedder.embed_query(question)
            hits = dense_search(self.conn, self.variant, vector, k=k)
        return [RankedChunk(hit.to_retrieved_item()) for hit in hits]


class HybridRetriever(_IndexRetriever):
    """Dense + Turkish full-text search over one chunk table, fused with RRF."""

    def __init__(
        self,
        conn: Connection,
        variant: str,
        embedder: Any,
        *,
        candidates_per_arm: int,
        rrf_k: int,
    ) -> None:
        super().__init__(conn, variant, embedder)
        self.candidates_per_arm = candidates_per_arm
        self.rrf_k = rrf_k

    def retrieve(self, question: str, k: int) -> list[RankedChunk]:
        with self._lock:
            vector = self.embedder.embed_query(question)
            arms = {
                "dense": dense_search(self.conn, self.variant, vector, k=self.candidates_per_arm),
                "fts": fts_search(self.conn, self.variant, question, k=self.candidates_per_arm),
            }
        return fuse_arms(arms, self.rrf_k)[:k]


def fuse_arms(arms: dict[str, Sequence[SearchHit]], rrf_k: int) -> list[RankedChunk]:
    """RRF over named ranked lists; each item's score is its fused score and its trace
    holds its 1-based rank in every arm (`<arm>_rank`, None when the arm missed it)."""
    ranks = {
        name: {hit.chunk_id: rank for rank, hit in enumerate(hits, start=1)}
        for name, hits in arms.items()
    }
    return [
        RankedChunk(
            dataclasses.replace(hit, score=score).to_retrieved_item(),
            {f"{name}_rank": ranks[name].get(hit.chunk_id) for name in arms},
        )
        for hit, score in rrf_fuse(list(arms.values()), k=rrf_k)
    ]


def _index_resources(config: RagConfig) -> tuple[Connection, str, Any]:
    """Connect to `DATABASE_URL` and load the embedder (needs the `ml` group)."""
    from kaprag.index.embed import Embedder
    from kaprag.index.store import connect

    embedder = Embedder(config.embedding.model, device=config.embedding.device)
    conn = connect()
    conn.autocommit = True
    return conn, config.index, embedder


def _open_dense(config: RagConfig) -> DenseRetriever:
    return DenseRetriever(*_index_resources(config))


def _open_hybrid(config: RagConfig) -> HybridRetriever:
    return HybridRetriever(
        *_index_resources(config),
        candidates_per_arm=config.retrieval.candidates_per_arm,
        rrf_k=config.retrieval.rrf_k,
    )


RETRIEVAL_MODES: dict[str, Callable[[RagConfig], Retriever]] = {
    "dense": _open_dense,
    "hybrid": _open_hybrid,
}


def open_retriever(config: RagConfig) -> Retriever:
    """The configured first-stage mode, wrapped in the reranking stage when the config
    has a `rerank` block. Models load here, once per run."""
    retriever = RETRIEVAL_MODES[config.retrieval.mode](config)
    if config.rerank is None:
        return retriever
    from kaprag.rag import rerank

    try:
        scorer = rerank.CrossEncoderScorer(
            config.rerank.model,
            device=config.rerank.device,
            batch_size=config.rerank.batch_size,
            max_length=config.rerank.max_length,
        )
    except BaseException:
        retriever.close()
        raise
    return rerank.RerankingRetriever(retriever, scorer, config.rerank.candidates)
