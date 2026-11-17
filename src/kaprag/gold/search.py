"""Corpus search for the probe and abstain stages: dense + full-text fused with RRF.

`Searcher` is the interface the stages use (tests pass a fake). `HybridSearcher` runs
both arms of `kaprag.retrieval` over one chunk table, optionally restricted to some
documents, and fuses them with `rrf_fuse`. HNSW iterative scan is switched on for the
session so a document filter still yields enough dense candidates.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from psycopg import Connection, sql

from kaprag.retrieval import SearchHit, dense_search, fts_search, rrf_fuse

ARM_K = 30


class Searcher(Protocol):
    def search(
        self, query: str, doc_ids: Sequence[str] | None = None, k: int = 8
    ) -> list[SearchHit]: ...


class HybridSearcher:
    def __init__(self, conn: Connection, variant: str, embedder: Any, arm_k: int = ARM_K) -> None:
        self.conn = conn
        self.variant = variant
        self.embedder = embedder
        self.arm_k = arm_k
        conn.execute("SET hnsw.iterative_scan = strict_order")

    def search(
        self, query: str, doc_ids: Sequence[str] | None = None, k: int = 8
    ) -> list[SearchHit]:
        where = sql.SQL("doc_id = ANY({})").format(sql.Literal(list(doc_ids))) if doc_ids else None
        vector = self.embedder.embed_query(query)
        dense = dense_search(self.conn, self.variant, vector, k=self.arm_k, where=where)
        fts = fts_search(self.conn, self.variant, query, k=self.arm_k, where=where)
        return [hit for hit, _ in rrf_fuse([dense, fts])[:k]]


def open_searcher(variant: str, device: str = "cuda") -> HybridSearcher:
    """Searcher over the `DATABASE_URL` index with the configured embedding model (needs
    the `ml` group; the model is loaded here)."""
    from kaprag.index.embed import Embedder, embed_model_name
    from kaprag.index.store import connect

    conn = connect()
    conn.autocommit = True
    return HybridSearcher(conn, variant, Embedder(embed_model_name(), device=device))


def table_complete(variant: str, doc_ids: Sequence[str]) -> dict[str, int]:
    """Row counts of the variant's table for `doc_ids` (missing documents count 0)."""
    from kaprag.index.store import connect, row_counts

    with connect() as conn:
        counts = row_counts(conn, variant)
    return {d: counts.get(d, 0) for d in doc_ids}
