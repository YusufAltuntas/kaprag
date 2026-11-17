"""The real cross-encoder reranker on the GPU (downloads the model on first use)."""

import pytest

pytestmark = pytest.mark.gpu


def test_reranker_ranks_the_relevant_passage_higher():
    from kaprag.rag.rerank import CrossEncoderScorer

    scorer = CrossEncoderScorer()
    relevant = (
        "Akbank'ın 2024 yılında konsolide net kârı 42,4 milyar TL olarak gerçekleşti; "
        "özkaynak kârlılığı yüzde 24 seviyesinde oldu."
    )
    unrelated = (
        "Sürdürülebilirlik stratejimiz kapsamında yenilenebilir enerji projelerine "
        "kredi desteği sağladık ve karbon emisyonlarımızı azalttık."
    )
    scores = scorer.score("Akbank 2024 konsolide net kâr ne kadar?", [unrelated, relevant])
    assert len(scores) == 2
    assert scores[1] > scores[0]
    assert scorer.score("soru", []) == []
