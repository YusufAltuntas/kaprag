"""Retrieval modes and their construction from the config.

A `Retriever` returns the top-k chunks for a question, best first, as the harness's
`RetrievedItem`s (text included). It is synchronous and may block (GPU embedding, a
database query); `kaprag.rag.pipeline` runs it in a worker thread.

`RETRIEVAL_MODES` maps a config `retrieval.mode` to an opener. Only `dense` exists
(the baseline). A new mode (`hybrid`) is one new opener here; a reranker or a SQL
filter wraps or parameterises a retriever here. Generation never changes for them.

Concurrency: a `DenseRetriever` owns one embedder and one autocommit connection,
neither safe for concurrent use, so `retrieve` holds a lock: retrievals are
serialised (each is tens of milliseconds next to seconds of generation).
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol

from kaprag.eval.pipeline import RetrievedItem
from kaprag.retrieval import dense_search

if TYPE_CHECKING:
    from psycopg import Connection

    from kaprag.rag.config import RagConfig


class Retriever(Protocol):
    def retrieve(self, question: str, k: int) -> list[RetrievedItem]: ...

    def close(self) -> None: ...


class DenseRetriever:
    """Dense (pgvector cosine) search over one chunk table."""

    def __init__(self, conn: Connection, variant: str, embedder: Any) -> None:
        self.conn = conn
        self.variant = variant
        self.embedder = embedder
        self._lock = threading.Lock()

    def retrieve(self, question: str, k: int) -> list[RetrievedItem]:
        with self._lock:
            vector = self.embedder.embed_query(question)
            hits = dense_search(self.conn, self.variant, vector, k=k)
        return [hit.to_retrieved_item() for hit in hits]

    def close(self) -> None:
        with self._lock:
            self.conn.close()


def _open_dense(config: RagConfig) -> DenseRetriever:
    """Connect to `DATABASE_URL` and load the embedder (needs the `ml` group)."""
    from kaprag.index.embed import Embedder
    from kaprag.index.store import connect

    embedder = Embedder(config.embedding.model, device=config.embedding.device)
    conn = connect()
    conn.autocommit = True
    return DenseRetriever(conn, config.index, embedder)


RETRIEVAL_MODES: dict[str, Callable[[RagConfig], Retriever]] = {
    "dense": _open_dense,
}


def open_retriever(config: RagConfig) -> Retriever:
    return RETRIEVAL_MODES[config.retrieval.mode](config)
