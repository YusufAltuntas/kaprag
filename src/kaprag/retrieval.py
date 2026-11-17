"""Search primitives over one chunk table: dense (pgvector HNSW) and Turkish full-text.

Both return the top-k `SearchHit`s best first; `SearchHit.to_retrieved_item()` gives
the eval harness's `RetrievedItem`. `rrf_fuse` merges ranked lists (reciprocal rank
fusion); the hybrid retrieval mode around it lives in `kaprag.rag.retriever`.

Filters: both functions take an optional `where`, ANDed into the WHERE clause in
`_search`, so it applies before ranking in both arms. It is either a `SqlFilter`
(conditions with bound parameters; what the RAG pipeline uses) or a bare
`psycopg.sql` Composable (e.g. `sql.SQL("bank = {}").format(sql.Literal("akbank"))`).
With the HNSW index a selective filter would return fewer than k rows, because the
index scan yields only `hnsw.ef_search` candidates before filtering; so a filtered
dense query turns on pgvector's iterative index scan (`hnsw.iterative_scan =
strict_order`: the scan continues until k rows pass the filter, results stay in exact
distance order, up to `hnsw.max_scan_tuples` = 20000 visited tuples by default). The
setting names and values are those of pgvector 0.8 (`pg_settings` of the installed
0.8.7: `hnsw.iterative_scan` enum off | relaxed_order | strict_order). Unfiltered
queries do not touch it.

Dense: cosine distance via `<=>` on normalised vectors, score = 1 - distance (cosine
similarity). `hnsw.ef_search` is raised to k for the query when k > 40 so the index
can return k rows (HNSW yields at most ef_search rows); the setting is local to the
query's transaction, so it works on autocommit connections and never leaks.

Full-text: a plain AND of all question terms (`plainto_tsquery`) matches almost no
chunk for a natural-language question, so the query is the OR of the question's
lexemes after the `turkish` configuration's stop-word removal and stemming
(`to_tsvector('turkish', question)`), built as a tsquery literal. Ranking is
`ts_rank(tsv, query, 1)`: chunks matching more (and more frequent) query lexemes rank
higher, normalised by log of the chunk length. Postgres ranking has no IDF, so very
common lexemes (e.g. "banka") weigh as much as rare ones.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from pgvector import Vector
from psycopg import Connection, sql

from kaprag.eval.pipeline import RetrievedItem
from kaprag.index.store import table_name

DEFAULT_HNSW_EF_SEARCH = 40
RRF_K = 60


@dataclass(frozen=True)
class SqlFilter:
    """ANDed SQL conditions on chunk-table columns with their bound parameters.

    Immutable; each method returns a new filter with one more condition. Values are
    never put into the SQL text: each condition refers to a named placeholder
    (`%(f0)s`, `%(f1)s`, ...) bound at execution. Column names are quoted identifiers
    chosen by code, never by model output. A new kind of condition (e.g. an access-role
    overlap, `access_roles && %(fN)s`) is one more method calling `_add`.
    """

    conditions: tuple[sql.Composable, ...] = ()
    params: Mapping[str, Any] = field(default_factory=dict)

    def any_of(self, column: str, values: Sequence[Any]) -> SqlFilter:
        """`column = ANY(values)`: the SQL `column IN (values)` as one array parameter."""
        return self._add("{column} = ANY({value})", column, values)

    def _add(self, template: str, column: str, values: Sequence[Any]) -> SqlFilter:
        if not values:
            raise ValueError(f"filter on {column!r} with no values would match nothing")
        name = f"f{len(self.params)}"
        condition = sql.SQL(template).format(
            column=sql.Identifier(column), value=sql.Placeholder(name)
        )
        return SqlFilter((*self.conditions, condition), {**self.params, name: list(values)})

    @property
    def condition(self) -> sql.Composable:
        """The conditions joined with AND (empty SQL when there are none)."""
        return sql.SQL(" AND ").join(self.conditions)

    def __bool__(self) -> bool:
        return bool(self.conditions)


Where = SqlFilter | sql.Composable


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
    where: Where | None = None,
) -> list[SearchHit]:
    """Top-k chunks by cosine similarity to `query_embedding` (normalised, same model
    as the index; see `kaprag.index.embed.Embedder.embed_query`). `conn` needs the
    vector adapter (`kaprag.index.store.connect`)."""
    vector = Vector(query_embedding)
    # `set_config(..., true)` is transaction-local: on an autocommit connection each
    # statement is its own transaction, so the setting and the query share an explicit
    # one (a savepoint when a transaction is already open).
    with conn.transaction():
        if k > DEFAULT_HNSW_EF_SEARCH:
            conn.execute("SELECT set_config('hnsw.ef_search', %s, true)", (str(k),))
        if _where_parts(where)[0] is not None:
            conn.execute("SELECT set_config('hnsw.iterative_scan', 'strict_order', true)")
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
    where: Where | None = None,
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
    where: Where | None,
) -> list[SearchHit]:
    conditions = [match] if match is not None else []
    where_sql, where_params = _where_parts(where)
    if where_sql is not None:
        conditions.append(sql.SQL("({})").format(where_sql))
    query = sql.SQL(
        "SELECT chunk_id, doc_id, page_start, page_end, {score} AS score, chunk_type,"
        " section_path, text FROM {table}{where} ORDER BY {order}, chunk_id LIMIT %(k)s"
    ).format(
        score=score,
        table=sql.Identifier(table_name(variant)),
        where=sql.SQL(" WHERE ") + sql.SQL(" AND ").join(conditions) if conditions else sql.SQL(""),
        order=order,
    )
    clash = set(where_params) & {*params, "k"}
    if clash:
        raise ValueError(f"filter parameters {sorted(clash)} clash with the query's")
    rows = conn.execute(query, {**params, **where_params, "k": k}).fetchall()
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


def _where_parts(where: Where | None) -> tuple[sql.Composable | None, Mapping[str, Any]]:
    """(condition or None, bound parameters) of a `where` argument."""
    if isinstance(where, SqlFilter):
        return (where.condition, where.params) if where else (None, {})
    return where, {}


def rrf_fuse(
    rankings: Sequence[Sequence[SearchHit]], k: int = RRF_K
) -> list[tuple[SearchHit, float]]:
    """Reciprocal rank fusion: score(chunk) = sum over lists of 1 / (k + rank), rank
    1-based. Returns (hit, fused score) best first, ties by chunk id; a chunk found
    in several lists keeps the hit object of the first list it appears in."""
    scores: dict[str, float] = {}
    hits: dict[str, SearchHit] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            hits.setdefault(hit.chunk_id, hit)
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (k + rank)
    order = sorted(scores, key=lambda cid: (-scores[cid], cid))
    return [(hits[cid], scores[cid]) for cid in order]
