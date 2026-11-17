import pytest

pytestmark = pytest.mark.gpu


def test_bge_m3_embeds_normalised_1024_dim_vectors():
    import numpy as np

    from kaprag.index.embed import Embedder

    embedder = Embedder("BAAI/bge-m3", batch_size=2)
    vectors = embedder.embed_documents(["Net kâr 2023 yılında arttı.", "Kredi kartı"])
    query = embedder.embed_query("net kâr")
    assert vectors.shape == (2, 1024)
    assert query.shape == (1024,)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-3)
    assert float(vectors[0] @ query) > float(vectors[1] @ query)
