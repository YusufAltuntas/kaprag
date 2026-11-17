"""Dense and hybrid retrieval for a real question against the built `chunks_fixed` index."""

import pytest

pytestmark = [pytest.mark.db, pytest.mark.gpu]


def test_dense_retriever_returns_ranked_fixed_chunks():
    from kaprag.rag.config import parse_rag_config
    from kaprag.rag.retriever import open_retriever

    config = parse_rag_config({"index": "fixed"})
    retriever = open_retriever(config)
    try:
        ranked = retriever.retrieve("Akbank'ın 2024 yılı konsolide net kârı ne kadar?", 10)
        items = [r.item for r in ranked]
    finally:
        retriever.close()

    assert len(items) == 10
    assert all(":fixed:" in i.chunk_id and i.text for i in items)
    scores = [i.score for i in items]
    assert scores == sorted(scores, reverse=True)
    assert any(i.doc.startswith("akbank-") for i in items[:5])


def test_hybrid_retriever_fuses_both_arms_over_fixed_chunks():
    from kaprag.rag.config import parse_rag_config
    from kaprag.rag.retriever import open_retriever

    config = parse_rag_config({"index": "fixed", "retrieval": {"mode": "hybrid"}})
    retriever = open_retriever(config)
    try:
        ranked = retriever.retrieve("Akbank'ın 2024 yılı konsolide net kârı ne kadar?", 40)
    finally:
        retriever.close()

    assert len(ranked) == 40
    assert all(":fixed:" in r.item.chunk_id and r.item.text for r in ranked)
    scores = [r.item.score for r in ranked]
    assert scores == sorted(scores, reverse=True)
    assert all(r.trace["dense_rank"] or r.trace["fts_rank"] for r in ranked)
    assert any(r.trace["dense_rank"] and r.trace["fts_rank"] for r in ranked)  # found by both
    assert any(r.trace["dense_rank"] is None for r in ranked)  # FTS-only items kept
    assert any(r.trace["fts_rank"] is None for r in ranked)  # dense-only items kept
    assert any(r.item.doc.startswith("akbank-") for r in ranked[:5])
