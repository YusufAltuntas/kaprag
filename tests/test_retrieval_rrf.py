"""Reciprocal rank fusion (pure; no database)."""

import pytest

from kaprag.retrieval import SearchHit, rrf_fuse


def hit(chunk_id: str, score: float = 1.0) -> SearchHit:
    return SearchHit(chunk_id, "akbank-2024", 1, 1, score, "text", (), chunk_id)


def test_rrf_sums_reciprocal_ranks_with_k60():
    dense = [hit("a"), hit("b"), hit("c")]
    fts = [hit("c"), hit("a")]
    fused = rrf_fuse([dense, fts])
    assert [h.chunk_id for h, _ in fused] == ["a", "c", "b"]
    scores = dict((h.chunk_id, s) for h, s in fused)
    assert scores["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert scores["c"] == pytest.approx(1 / 63 + 1 / 61)
    assert scores["b"] == pytest.approx(1 / 62)


def test_rrf_keeps_first_seen_hit_and_breaks_ties_by_chunk_id():
    fused = rrf_fuse([[hit("b", 0.9)], [hit("a", 0.5)]])
    assert [h.chunk_id for h, _ in fused] == ["a", "b"]
    again = rrf_fuse([[hit("x", 0.9)], [hit("x", 0.1)]])
    assert again[0][0].score == 0.9


def test_rrf_custom_k_and_empty():
    assert rrf_fuse([]) == []
    fused = rrf_fuse([[hit("a")]], k=0)
    assert fused[0][1] == 1.0
