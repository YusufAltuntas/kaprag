"""Dense retrieval for a real question against the built `chunks_fixed` index."""

import pytest

pytestmark = [pytest.mark.db, pytest.mark.gpu]


def test_dense_retriever_returns_ranked_fixed_chunks():
    from kaprag.rag.config import parse_rag_config
    from kaprag.rag.retriever import open_retriever

    config = parse_rag_config({"index": "fixed"})
    retriever = open_retriever(config)
    try:
        items = retriever.retrieve("Akbank'ın 2024 yılı konsolide net kârı ne kadar?", 10)
    finally:
        retriever.close()

    assert len(items) == 10
    assert all(":fixed:" in i.chunk_id and i.text for i in items)
    scores = [i.score for i in items]
    assert scores == sorted(scores, reverse=True)
    assert any(i.doc.startswith("akbank-") for i in items[:5])
