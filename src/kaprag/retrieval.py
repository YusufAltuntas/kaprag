"""Search primitives over one chunk table: dense (pgvector HNSW) and Turkish full-text.

Both return the top-k `SearchHit`s best first; `SearchHit.to_retrieved_item()` gives
the eval harness's `RetrievedItem`. Fusion and reranking are composed by later steps.

Filters: both functions take an optional `where` SQL condition (a `psycopg.sql`
Composable, e.g. `sql.SQL("bank = {}").format(sql.Literal("akbank"))`). It is ANDed
into the WHERE clause in `_search`, so it applies before ranking in both arms. (With
the HNSW index a selective filter can return fewer than k rows, because the index scan
yields `hnsw.ef_search` candidates before filtering; the task adding filters should
enable pgvector's `hnsw.iterative_scan` or raise ef_search.)

Dense: cosine distance via `<=>` on normalised vectors, score = 1 - distance (cosine
similarity). `hnsw.ef_search` is raised to k for the query when k > 40 so the index
can return k rows.

Full-text: a plain AND of all question terms (`plainto_tsquery`) matches almost no
chunk for a natural-language question, so the query is the OR of the question's
lexemes after the `turkish` configuration's stop-word removal and stemming
(`to_tsvector('turkish', question)`), built as a tsquery literal. Ranking is
`ts_rank(tsv, query, 1)`: chunks matching more (and more frequent) query lexemes rank
higher, normalised by log of the chunk length. Postgres ranking has no IDF, so very
common lexemes (e.g. "banka") weigh as much as rare ones.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from pgvector import Vector
from psycopg import Connection, sql

from kaprag.eval.pipeline import RetrievedItem
from kaprag.index.store import table_name

DEFAULT_HNSW_EF_SEARCH = 40


@dataclass(frozen=True)
class SearchHit:
    chunk_id: str
    doc_id: str
    page_start: int
    page_end: int
    score: float
    chunk_type: str
    section_path: tuple[str, ...]
    text: str

    def to_retrieved_item(self) -> RetrievedItem:
        return RetrievedItem(
            chunk_id=self.chunk_id,
            doc=self.doc_id,
            page_start=self.page_start,
            page_end=self.page_end,
            score=self.score,
            text=self.text,
        )


def dense_search(
    conn: Connection,
    variant: str,
    query_embedding: Sequence[float] | Any,
    k: int = 10,
    where: sql.Composable | None = None,
) -> list[SearchHit]:
    """Top-k chunks by cosine similarity to `query_embedding` (normalised, same model
    as the index; see `kaprag.index.embed.Embedder.embed_query`). `conn` needs the
    vector adapter (`kaprag.index.store.connect`)."""
    vector = Vector(query_embedding)
    if k > DEFAULT_HNSW_EF_SEARCH:
        conn.execute("SELECT set_config('hnsw.ef_search', %s, true)", (str(k),))
    return _search(
        conn,
        variant,
        score=sql.SQL("1 - (embedding <=> %(q)s)"),
        match=None,
        order=sql.SQL("embedding <=> %(q)s"),
        params={"q": vector},
        k=k,
        where=where,
    )


def fts_search(
    conn: Connection,
    variant: str,
    query: str,
    k: int = 10,
    where: sql.Composable | None = None,
) -> list[SearchHit]:
    """Top-k chunks by Turkish full-text rank for the OR of the query's lexemes."""
    tsquery = or_tsquery(query_lexemes(conn, query))
    if tsquery is None:
        return []
    return _search(
        conn,
        variant,
        score=sql.SQL("ts_rank(tsv, %(q)s::tsquery, 1)"),
        match=sql.SQL("tsv @@ %(q)s::tsquery"),
        order=sql.SQL("ts_rank(tsv, %(q)s::tsquery, 1) DESC"),
        params={"q": tsquery},
        k=k,
        where=where,
    )


def query_lexemes(conn: Connection, query: str) -> list[str]:
    """The query's lexemes under the `turkish` configuration (stop words removed,
    stemmed), in order of first occurrence."""
    rows = conn.execute(
        "SELECT lexeme FROM unnest(to_tsvector('turkish', %s)) ORDER BY positions[1]",
        (query,),
    ).fetchall()
    return [lexeme for (lexeme,) in rows]


def or_tsquery(lexemes: Sequence[str]) -> str | None:
    """A tsquery literal ORing already-normalised lexemes, or None if there are none."""
    if not lexemes:
        return None
    return " | ".join(_quote_lexeme(x) for x in lexemes)


def _quote_lexeme(lexeme: str) -> str:
    escaped = lexeme.replace("\\", "\\\\").replace("'", "''")
    return f"'{escaped}'"


def _search(
    conn: Connection,
    variant: str,
    *,
    score: sql.Composable,
    match: sql.Composable | None,
    order: sql.Composable,
    params: dict[str, Any],
    k: int,
    where: sql.Composable | None,
) -> list[SearchHit]:
    conditions = [match] if match is not None else []
    if where is not None:
        conditions.append(sql.SQL("({})").format(where))
    query = sql.SQL(
        "SELECT chunk_id, doc_id, page_start, page_end, {score} AS score, chunk_type,"
        " section_path, text FROM {table}{where} ORDER BY {order}, chunk_id LIMIT %(k)s"
    ).format(
        score=score,
        table=sql.Identifier(table_name(variant)),
        where=sql.SQL(" WHERE ") + sql.SQL(" AND ").join(conditions) if conditions else sql.SQL(""),
        order=order,
    )
    rows = conn.execute(query, {**params, "k": k}).fetchall()
    return [
        SearchHit(
            chunk_id=chunk_id,
            doc_id=doc_id,
            page_start=page_start,
            page_end=page_end,
            score=float(score),
            chunk_type=chunk_type,
            section_path=tuple(section_path),
            text=text,
        )
        for chunk_id, doc_id, page_start, page_end, score, chunk_type, section_path, text in rows
    ]
