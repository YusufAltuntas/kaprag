"""Hybrid retrieval mode: dense + Turkish FTS fused with RRF (fake arms; no GPU or DB)."""

import pytest

from kaprag.rag import retriever as retriever_module
from kaprag.rag.config import parse_rag_config
from kaprag.rag.retriever import RETRIEVAL_MODES, HybridRetriever, RankedChunk
from kaprag.retrieval import SearchHit


def hit(chunk_id: str, score: float = 0.5) -> SearchHit:
    return SearchHit(chunk_id, "akbank-2024", 3, 4, score, "text", (), f"text of {chunk_id}")


class FakeEmbedder:
    def __init__(self):
        self.queries = []

    def embed_query(self, text):
        self.queries.append(text)
        return [0.1, 0.2]


class FakeConn:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


@pytest.fixture
def arms(monkeypatch):
    """Scripted dense and FTS arms recording how they were called."""
    calls = {"dense": [], "fts": []}
    results = {
        "dense": [hit("a", 0.9), hit("b", 0.8), hit("c", 0.7)],
        "fts": [hit("c", 12.0), hit("d", 3.0), hit("a", 1.0)],
    }

    def dense_search(conn, variant, vector, k=10, where=None):
        calls["dense"].append((variant, list(vector), k, where))
        return results["dense"][:k]

    def fts_search(conn, variant, query, k=10, where=None):
        calls["fts"].append((variant, query, k, where))
        return results["fts"][:k]

    monkeypatch.setattr(retriever_module, "dense_search", dense_search)
    monkeypatch.setattr(retriever_module, "fts_search", fts_search)
    return calls, results


def make(candidates_per_arm=40, rrf_k=60):
    embedder = FakeEmbedder()
    retriever = HybridRetriever(
        FakeConn(), "fixed", embedder, candidates_per_arm=candidates_per_arm, rrf_k=rrf_k
    )
    return retriever, embedder


def test_both_arms_are_called_with_the_candidate_depth(arms):
    calls, _ = arms
    retriever, embedder = make(candidates_per_arm=40)
    retriever.retrieve("Akbank net kâr?", 10)
    assert embedder.queries == ["Akbank net kâr?"]
    assert calls["dense"] == [("fixed", [0.1, 0.2], 40, None)]
    assert calls["fts"] == [("fixed", "Akbank net kâr?", 40, None)]


def test_fused_order_score_and_per_arm_ranks(arms):
    retriever, _ = make()
    ranked = retriever.retrieve("soru", 10)

    # a: 1/61 + 1/63, c: 1/63 + 1/61 (tie -> chunk id), b: 1/62, d: 1/62 (tie -> chunk id)
    assert [r.item.chunk_id for r in ranked] == ["a", "c", "b", "d"]
    assert all(isinstance(r, RankedChunk) for r in ranked)
    assert ranked[0].item.score == pytest.approx(1 / 61 + 1 / 63)
    assert ranked[2].item.score == pytest.approx(1 / 62)
    assert [r.trace for r in ranked] == [
        {"dense_rank": 1, "fts_rank": 3},
        {"dense_rank": 3, "fts_rank": 1},
        {"dense_rank": 2, "fts_rank": None},
        {"dense_rank": None, "fts_rank": 2},
    ]
    # the item is the harness's RetrievedItem, text included
    assert (ranked[3].item.doc, ranked[3].item.page_start, ranked[3].item.text) == (
        "akbank-2024",
        3,
        "text of d",
    )


def test_single_arm_items_are_kept_when_the_other_arm_is_empty(arms):
    _, results = arms
    results["fts"] = []  # e.g. a question of only stop words
    retriever, _ = make()
    ranked = retriever.retrieve("ve ile", 10)
    assert [r.item.chunk_id for r in ranked] == ["a", "b", "c"]
    assert [r.trace["fts_rank"] for r in ranked] == [None, None, None]


def test_retrieve_k_is_respected(arms):
    retriever, _ = make()
    assert [r.item.chunk_id for r in retriever.retrieve("soru", 2)] == ["a", "c"]


def test_rrf_k_comes_from_the_config(arms):
    retriever, _ = make(rrf_k=0)
    ranked = retriever.retrieve("soru", 10)
    assert ranked[0].item.score == pytest.approx(1 / 1 + 1 / 3)


def test_close_closes_the_connection(arms):
    retriever, _ = make()
    retriever.close()
    assert retriever.conn.closed


def test_hybrid_is_a_registered_mode():
    assert "hybrid" in RETRIEVAL_MODES
    config = parse_rag_config({"index": "fixed", "retrieval": {"mode": "hybrid"}})
    assert (config.retrieval.candidates_per_arm, config.retrieval.rrf_k) == (40, 60)
