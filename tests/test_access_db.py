"""Access filtering and role assignment against Postgres (throwaway schema)."""

import os
import random
import uuid

import pytest
from psycopg import sql

from kaprag.access.assign import apply_roles, distribution, document_roles
from kaprag.access.filter import access_filter
from kaprag.access.roles import FOOTNOTES, PUBLIC, RISK, can_see
from kaprag.access.rules import DocumentAccess
from kaprag.chunking.base import Chunk
from kaprag.corpus import Document
from kaprag.index.store import EMBEDDING_DIM, connect, ensure_schema, replace_document
from kaprag.rag.query_filter import metadata_filter
from kaprag.retrieval import dense_search, fts_search

pytestmark = pytest.mark.db

ROWS_PER_DOC = 40
BANKS = ("garanti", "akbank", "isbank", "yapikredi")
YEARS = (2023, 2024, 2025)


@pytest.fixture
def schema():
    name = f"kaprag_test_{uuid.uuid4().hex[:12]}"
    with connect(os.environ["DATABASE_URL"]) as c:
        c.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(name)))
        c.execute(sql.SQL("SET search_path TO {}, public").format(sql.Identifier(name)))
        c.commit()
        ensure_schema(c)
        try:
            yield name
        finally:
            c.rollback()
            c.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(name)))
            c.commit()


@pytest.fixture
def conn(schema):
    with connect(os.environ["DATABASE_URL"], schema=schema) as c:
        yield c


def doc(doc_id: str) -> Document:
    bank, year = doc_id.split("-")
    return Document(doc_id, bank, int(year), "https://example.test", None, None, "")


def page_role(page: int) -> frozenset[str]:
    """Pages 1, 6, 11, ... risk; 2, 7, 12, ... footnotes; the rest public (40% restricted)."""
    return {1: RISK, 2: FOOTNOTES}.get(page % 5, PUBLIC)


ACCESS = DocumentAccess(
    page_roles={p: page_role(p) for p in range(1, ROWS_PER_DOC + 1)},
    section_roles={p: page_role(p) for p in range(1, ROWS_PER_DOC + 1)},
)


def chunk(doc_id: str, ordinal: int, text: str) -> Chunk:
    page = ordinal + 1  # one page per chunk, so its role is page_role(page)
    return Chunk(doc_id, ordinal, page, page, "text", ("Bölüm",), text, 3)


def random_unit(rng: random.Random) -> list[float]:
    v = [rng.gauss(0.0, 1.0) for _ in range(EMBEDDING_DIM)]
    norm = sum(x * x for x in v) ** 0.5
    return [x / norm for x in v]


def fill(conn, access: DocumentAccess = ACCESS, variant: str = "fixed") -> None:
    rng = random.Random(7)
    for bank in BANKS:
        for year in YEARS:
            d = doc(f"{bank}-{year}")
            chunks = [
                chunk(d.doc_id, i, f"risk yönetimi likidite {bank} {year} {i}")
                for i in range(ROWS_PER_DOC)
            ]
            vectors = [random_unit(rng) for _ in chunks]
            replace_document(conn, variant, d, chunks, vectors, access=access)
    conn.commit()


def roles_of(conn, ids) -> dict[str, list[str]]:
    rows = conn.execute(
        "SELECT chunk_id, access_roles FROM chunks_fixed WHERE chunk_id = ANY(%s)", (list(ids),)
    ).fetchall()
    return dict(rows)


def test_replace_document_assigns_roles_from_the_page_map(conn):
    fill(conn)
    assert distribution(
        [r for (r,) in conn.execute("SELECT access_roles FROM chunks_fixed").fetchall()]
    ) == {"public": 288, "analyst,risk": 96, "risk": 96}
    assert roles_of(conn, ["akbank-2024:fixed:0", "akbank-2024:fixed:1"]) == {
        "akbank-2024:fixed:0": ["risk"],
        "akbank-2024:fixed:1": ["analyst", "risk"],
    }


@pytest.mark.parametrize("role", ["public", "analyst", "risk"])
def test_filtered_dense_and_fts_return_only_visible_rows_and_the_requested_count(conn, role):
    fill(conn)
    conn.autocommit = True
    conn.execute("SET enable_seqscan = off")  # force the HNSW / GIN index paths
    query = random_unit(random.Random(99))
    where = access_filter(role)
    dense = dense_search(conn, "fixed", query, k=40, where=where)
    fts = fts_search(conn, "fixed", "risk yönetimi likidite", k=40, where=where)
    for hits in (dense, fts):
        assert len(hits) == 40
        roles = roles_of(conn, [h.chunk_id for h in hits])
        assert all(can_see(roles[h.chunk_id], role) for h in hits)
    unfiltered = dense_search(conn, "fixed", query, k=40)
    restricted = [
        cid for cid, r in roles_of(conn, [h.chunk_id for h in unfiltered]).items()
        if not can_see(r, "public")
    ]  # fmt: skip
    assert restricted  # without the filter, public would have seen restricted rows


def test_access_and_metadata_filters_combine(conn):
    fill(conn)
    conn.autocommit = True
    conn.execute("SET enable_seqscan = off")
    where = access_filter("public", metadata_filter(["akbank"], [2024]))
    query = random_unit(random.Random(5))
    dense = dense_search(conn, "fixed", query, k=24, where=where)
    fts = fts_search(conn, "fixed", "likidite", k=30, where=where)
    assert len(dense) == 24  # akbank-2024 has 24 public rows
    assert len(fts) == 24  # asked for more than exist: all of them, nothing else
    for hits in (dense, fts):
        assert {h.doc_id for h in hits} == {"akbank-2024"}
        assert all(r == ["public"] for r in roles_of(conn, [h.chunk_id for h in hits]).values())


def test_assignment_updates_only_access_roles_per_document_idempotently(conn):
    fill(conn, access=DocumentAccess())  # indexed before roles existed: all public
    columns = (
        "chunk_id, doc_id, ordinal, bank, year, page_start, page_end, section_path,"
        " chunk_type, text, embedding::text, tsv::text"
    )
    snapshot = f"SELECT {columns} FROM chunks_fixed ORDER BY chunk_id"
    before = conn.execute(snapshot).fetchall()

    rows = document_roles(conn, "fixed", "akbank-2024", ACCESS)
    assert len(rows) == ROWS_PER_DOC
    assert distribution([r.before for r in rows]) == {"public": 40}
    assert distribution([r.after for r in rows]) == {"public": 24, "analyst,risk": 8, "risk": 8}
    assert apply_roles(conn, "fixed", "akbank-2024", rows) == 16

    assert conn.execute(snapshot).fetchall() == before  # no other column changed
    by_doc = dict(
        conn.execute(
            "SELECT doc_id, count(*) FROM chunks_fixed WHERE access_roles <> '{public}'"
            " GROUP BY doc_id"
        ).fetchall()
    )
    assert by_doc == {"akbank-2024": 16}  # only that document

    again = document_roles(conn, "fixed", "akbank-2024", ACCESS)
    assert all(r.before == r.after for r in again)
    assert apply_roles(conn, "fixed", "akbank-2024", again) == 0  # idempotent


def test_connect_to_a_schema_requires_its_tables(schema):
    with connect(os.environ["DATABASE_URL"], schema=schema) as c:
        assert c.execute("SELECT current_schemas(false)").fetchone()[0][0] == schema
    with pytest.raises(ValueError, match="has no table"):
        connect(os.environ["DATABASE_URL"], schema="kaprag_no_such_schema")
    with pytest.raises(ValueError, match="invalid schema"):
        connect(os.environ["DATABASE_URL"], schema="x; DROP TABLE y")
