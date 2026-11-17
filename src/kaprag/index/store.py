"""Postgres/pgvector chunk tables: schema, per-document replace, row counts.

One table per chunking variant (`chunks_fixed`, `chunks_structure`), same shape:

    chunk_id      text PRIMARY KEY     -- '<doc_id>:<variant>:<ordinal>'
    doc_id        text                 -- with ordinal: UNIQUE (doc_id, ordinal)
    ordinal       integer              -- chunk position in the document, reading order
    bank, year                         -- from the manifest
    page_start, page_end  integer      -- 1-based PDF page indices, inclusive
    section_path  text[]               -- Docling headings (best effort); '{}' for fixed
    chunk_type    text                 -- see kaprag.chunking.base.CHUNK_TYPES
    access_roles  text[]               -- who may retrieve the chunk (simulated roles,
                                       -- see kaprag.access); set by access_roles_for
    text          text
    embedding     vector(1024)         -- normalised; HNSW index, cosine
    tsv           tsvector GENERATED ALWAYS AS (to_tsvector('turkish', text)) -- GIN index

The chunk id is stable across rebuilds as long as the chunker and its input are
unchanged (the ordinal is deterministic). Tables are created in the connection's
`search_path` (tests use a throwaway schema; `connect(schema=...)` selects another
schema, e.g. a scratch copy of the index).
"""

import os
import re
from collections.abc import Sequence
from typing import Any

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector
from psycopg import Connection, sql
from psycopg.pq import TransactionStatus

from kaprag.access.roles import role_list
from kaprag.access.rules import DocumentAccess, chunk_roles
from kaprag.chunking.base import Chunk
from kaprag.corpus import Document

EMBEDDING_DIM = 1024
VARIANTS = ("fixed", "structure")
_SCHEMA_NAME = re.compile(r"[a-z_][a-z0-9_]{0,62}")


def table_name(variant: str) -> str:
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}, expected one of {VARIANTS}")
    return f"chunks_{variant}"


def chunk_id(doc_id: str, variant: str, ordinal: int) -> str:
    return f"{doc_id}:{variant}:{ordinal}"


def access_roles_for(
    chunk: Chunk, doc: Document, variant: str, access: DocumentAccess
) -> list[str]:
    """Roles allowed to retrieve a chunk, assigned at indexing time by the SIMULATED
    access rule (`kaprag.access.rules`) from the document's per-page role map
    (`kaprag.index.build.load_document_access`)."""
    return role_list(
        chunk_roles(
            variant, chunk.section_path, chunk.chunk_type, chunk.page_start, chunk.page_end, access
        )
    )


_DDL = """
CREATE TABLE IF NOT EXISTS {table} (
    chunk_id     text PRIMARY KEY,
    doc_id       text NOT NULL,
    ordinal      integer NOT NULL,
    bank         text NOT NULL,
    year         integer NOT NULL,
    page_start   integer NOT NULL,
    page_end     integer NOT NULL,
    section_path text[] NOT NULL DEFAULT '{{}}',
    chunk_type   text NOT NULL,
    access_roles text[] NOT NULL DEFAULT '{{public}}',
    text         text NOT NULL,
    embedding    vector({dim}) NOT NULL,
    tsv          tsvector GENERATED ALWAYS AS (to_tsvector('turkish', text)) STORED,
    UNIQUE (doc_id, ordinal),
    CHECK (1 <= page_start AND page_start <= page_end)
);
CREATE INDEX IF NOT EXISTS {hnsw} ON {table} USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS {gin} ON {table} USING gin (tsv);
"""


def connect(dsn: str | None = None, *, schema: str | None = None) -> Connection:
    """Connect to `dsn` (default: `DATABASE_URL`) with the vector type adapter
    registered; the pgvector extension must exist (see `ensure_schema`).

    With `schema`, the connection's `search_path` is `<schema>, public` (set at
    connection start, so no rollback undoes it) and both chunk tables must exist in
    that schema, so a query can never silently fall through to the `public` tables.
    Used to run the scripts and the retriever against a scratch copy of the index.
    """
    options = None
    if schema is not None:
        if not _SCHEMA_NAME.fullmatch(schema):
            raise ValueError(f"invalid schema name {schema!r}")
        options = f"-c search_path={schema},public"
    conn = psycopg.connect(dsn or os.environ["DATABASE_URL"], connect_timeout=10, options=options)
    try:
        register_vector(conn)
        if schema is not None:
            missing = [
                table
                for table in map(table_name, VARIANTS)
                if conn.execute("SELECT to_regclass(%s)", (f"{schema}.{table}",)).fetchone()[0]
                is None
            ]
            conn.rollback()
            if missing:
                raise ValueError(f"schema {schema!r} has no table(s) {', '.join(missing)}")
    except BaseException:
        conn.close()
        raise
    return conn


def ensure_schema(conn: Connection, variants: Sequence[str] = VARIANTS) -> None:
    """Create the pgvector extension and the chunk tables with their indexes (idempotent).

    Commits, and registers the vector type adapter on `conn`.
    """
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    for variant in variants:
        table = table_name(variant)
        conn.execute(
            sql.SQL(_DDL).format(
                table=sql.Identifier(table),
                hnsw=sql.Identifier(f"{table}_embedding_hnsw"),
                gin=sql.Identifier(f"{table}_tsv_gin"),
                dim=sql.Literal(EMBEDDING_DIM),
            )
        )
    conn.commit()
    register_vector(conn)


def replace_document(
    conn: Connection,
    variant: str,
    doc: Document,
    chunks: Sequence[Chunk],
    embeddings: Sequence[Any],
    *,
    access: DocumentAccess,
) -> int:
    """Replace all rows of `doc` in the variant's table in one committed transaction;
    returns the number of rows written. `embeddings[i]` (list or numpy array) belongs
    to `chunks[i]`; `access` is the document's role map, from which each row's
    `access_roles` is assigned (`access_roles_for`).

    The write is always its own top-level transaction, committed on return: a
    transaction already open on `conn` (e.g. the implicit one a plain SELECT leaves
    behind) is committed first, so the write can never become a savepoint inside a
    longer transaction that a later rollback or crash would discard. A connection in a
    failed transaction is refused.
    """
    status = conn.info.transaction_status
    if status == TransactionStatus.INTRANS:
        conn.commit()
    elif status != TransactionStatus.IDLE:
        raise RuntimeError(f"connection not idle ({status.name}); roll back first")
    if len(chunks) != len(embeddings):
        raise ValueError(f"{len(chunks)} chunks but {len(embeddings)} embeddings")
    table = sql.Identifier(table_name(variant))
    rows = [
        (
            chunk_id(doc.doc_id, variant, c.ordinal),
            doc.doc_id,
            c.ordinal,
            doc.bank,
            doc.year,
            c.page_start,
            c.page_end,
            list(c.section_path),
            c.chunk_type,
            access_roles_for(c, doc, variant, access),
            c.text,
            Vector(e),
        )
        for c, e in zip(chunks, embeddings, strict=True)
    ]
    if foreign := {c.doc_id for c in chunks} - {doc.doc_id}:
        raise ValueError(f"chunks of {sorted(foreign)} written as {doc.doc_id!r}")
    with conn.transaction(), conn.cursor() as cur:
        cur.execute(sql.SQL("DELETE FROM {} WHERE doc_id = %s").format(table), (doc.doc_id,))
        cur.executemany(
            sql.SQL(
                "INSERT INTO {} (chunk_id, doc_id, ordinal, bank, year, page_start, page_end,"
                " section_path, chunk_type, access_roles, text, embedding)"
                " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
            ).format(table),
            rows,
        )
    return len(rows)


def row_counts(conn: Connection, variant: str) -> dict[str, int]:
    """Rows per doc_id in the variant's table, sorted by doc_id."""
    rows = conn.execute(
        sql.SQL("SELECT doc_id, count(*) FROM {} GROUP BY doc_id ORDER BY doc_id").format(
            sql.Identifier(table_name(variant))
        )
    ).fetchall()
    return {doc_id: n for doc_id, n in rows}
