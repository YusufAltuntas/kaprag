import os
import uuid

import pytest
from psycopg import sql

from kaprag.chunking.base import Chunk
from kaprag.corpus import Document
from kaprag.index.store import EMBEDDING_DIM, connect, ensure_schema, replace_document, row_counts
from kaprag.retrieval import dense_search, fts_search

pytestmark = pytest.mark.db


@pytest.fixture
def conn():
    """A connection whose search_path points at a throwaway schema with fresh tables."""
    schema = f"kaprag_test_{uuid.uuid4().hex[:12]}"
    with connect(os.environ["DATABASE_URL"]) as c:
        c.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
        c.execute(sql.SQL("SET search_path TO {}, public").format(sql.Identifier(schema)))
        c.commit()
        ensure_schema(c)
        try:
            yield c
        finally:
            c.rollback()
            c.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))
            c.commit()


def doc(doc_id: str) -> Document:
    bank, year = doc_id.split("-")
    return Document(doc_id, bank, int(year), "https://example.test", None, None, "")


def chunk(doc_id: str, ordinal: int, text: str, page: int = 1) -> Chunk:
    return Chunk(doc_id, ordinal, page, page + 1, "text", ("Bölüm",), text, 3)


def one_hot(i: int, weight: float = 1.0) -> list[float]:
    v = [0.0] * EMBEDDING_DIM
    v[i] = weight
    return v


def current_schema(conn) -> str:
    return conn.execute("SELECT current_schema()").fetchone()[0]


def test_schema_creation_is_idempotent_and_complete(conn):
    ensure_schema(conn)  # second time: no error
    schema = current_schema(conn)
    for table in ("chunks_fixed", "chunks_structure"):
        cols = dict(
            conn.execute(
                "SELECT column_name, is_generated FROM information_schema.columns"
                " WHERE table_schema = %s AND table_name = %s",
                (schema, table),
            ).fetchall()
        )
        assert set(cols) == {
            "chunk_id", "doc_id", "ordinal", "bank", "year", "page_start", "page_end",
            "section_path", "chunk_type", "access_roles", "text", "embedding", "tsv",
        }  # fmt: skip
        assert cols["tsv"] == "ALWAYS"
        indexes = " ".join(
            row[0]
            for row in conn.execute(
                "SELECT indexdef FROM pg_indexes WHERE schemaname = %s AND tablename = %s",
                (schema, table),
            ).fetchall()
        )
        assert "USING hnsw (embedding vector_cosine_ops)" in indexes
        assert "USING gin (tsv)" in indexes


def test_replace_document_replaces_only_that_document(conn):
    a, b = doc("akbank-2023"), doc("garanti-2023")
    replace_document(conn, "fixed", a, [chunk(a.doc_id, i, f"a{i}") for i in range(3)],
                     [one_hot(i) for i in range(3)])  # fmt: skip
    replace_document(conn, "fixed", b, [chunk(b.doc_id, i, f"b{i}") for i in range(2)],
                     [one_hot(i) for i in range(2)])  # fmt: skip
    assert row_counts(conn, "fixed") == {"akbank-2023": 3, "garanti-2023": 2}

    written = replace_document(conn, "fixed", a, [chunk(a.doc_id, 0, "new")], [one_hot(5)])
    assert written == 1
    assert row_counts(conn, "fixed") == {"akbank-2023": 1, "garanti-2023": 2}
    row = conn.execute(
        "SELECT chunk_id, bank, year, page_start, page_end, section_path, access_roles, text"
        " FROM chunks_fixed WHERE doc_id = 'akbank-2023'"
    ).fetchone()
    assert row == ("akbank-2023:fixed:0", "akbank", 2023, 1, 2, ["Bölüm"], ["public"], "new")
    assert row_counts(conn, "structure") == {}


def test_replace_document_rolls_back_on_error(conn):
    a = doc("akbank-2023")
    replace_document(conn, "fixed", a, [chunk(a.doc_id, 0, "keep")], [one_hot(0)])
    duplicate = [chunk(a.doc_id, 0, "x"), chunk(a.doc_id, 0, "y")]
    with pytest.raises(Exception, match="duplicate key"):
        replace_document(conn, "fixed", a, duplicate, [one_hot(0), one_hot(1)])
    assert conn.execute("SELECT text FROM chunks_fixed").fetchall() == [("keep",)]


def test_dense_search_ranks_by_cosine(conn):
    a = doc("akbank-2023")
    chunks = [chunk(a.doc_id, i, f"t{i}", page=i + 1) for i in range(3)]
    replace_document(conn, "structure", a, chunks, [one_hot(0), one_hot(1), one_hot(2)])
    query = one_hot(1, 0.8)
    query[2] = 0.6
    hits = dense_search(conn, "structure", query, k=2)
    assert [h.chunk_id for h in hits] == ["akbank-2023:structure:1", "akbank-2023:structure:2"]
    assert hits[0].score == pytest.approx(0.8, abs=1e-5)
    assert (hits[0].page_start, hits[0].page_end) == (2, 3)
    assert hits[0].section_path == ("Bölüm",)
    item = hits[0].to_retrieved_item()
    assert (item.doc, item.page_start, item.text) == ("akbank-2023", 2, "t1")


def test_fts_ranks_partial_matches_of_a_natural_question(conn):
    a = doc("akbank-2023")
    texts = [
        "Bankanın net kârı 2023 yılında yüzde 40 artarak 66 milyar TL oldu.",
        "Kredi kartı müşteri sayısı 2023 yılında arttı.",
        "Sürdürülebilirlik stratejimiz iklim riskini kapsar.",
    ]
    chunks = [chunk(a.doc_id, i, t) for i, t in enumerate(texts)]
    replace_document(conn, "fixed", a, chunks, [one_hot(i) for i in range(3)])

    hits = fts_search(conn, "fixed", "Akbank'ın 2023 yılı net kârı ne kadar oldu?", k=5)
    assert hits[0].text == texts[0]
    assert texts[2] not in [h.text for h in hits]  # no shared lexeme
    assert all(h.score > 0 for h in hits)
    assert hits == sorted(hits, key=lambda h: -h.score)


def test_fts_query_of_only_stop_words_returns_nothing(conn):
    a = doc("akbank-2023")
    replace_document(conn, "fixed", a, [chunk(a.doc_id, 0, "ve ile bu")], [one_hot(0)])
    assert fts_search(conn, "fixed", "ve ile bu", k=5) == []


def test_where_condition_applies_before_ranking(conn):
    a, b = doc("akbank-2023"), doc("garanti-2023")
    replace_document(conn, "fixed", a, [chunk(a.doc_id, 0, "net kâr")], [one_hot(0)])
    replace_document(conn, "fixed", b, [chunk(b.doc_id, 0, "net kâr arttı")], [one_hot(1)])
    where = sql.SQL("bank = {}").format(sql.Literal("akbank"))
    assert [h.doc_id for h in fts_search(conn, "fixed", "net kâr", where=where)] == ["akbank-2023"]
    assert [h.doc_id for h in dense_search(conn, "fixed", one_hot(1), where=where)] == [
        "akbank-2023"
    ]
