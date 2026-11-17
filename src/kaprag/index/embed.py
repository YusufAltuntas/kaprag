"""Dense embeddings with sentence-transformers (needs the `ml` group; imported lazily).

The model loads in fp16 on CUDA and encodes in small batches with a capped sequence
length, so a long chunk cannot blow up memory on a 4 GB GPU; vectors are L2-normalised
(cosine similarity = dot product). There is no silent CPU fallback: `device="cuda"`
fails when CUDA is unavailable, and CPU must be asked for explicitly.

Queries and documents go through separate methods so a model that needs prefixes
(e.g. "query: " / "passage: ") changes one place; bge-m3 uses none.
"""

import os
from collections.abc import Sequence
from typing import Any

DEFAULT_EMBED_MODEL = "BAAI/bge-m3"
DEFAULT_BATCH_SIZE = 8
DEFAULT_MAX_SEQ_LENGTH = 1024


def embed_model_name() -> str:
    """The configured embedding model: `EMBED_MODEL` from the environment, else bge-m3."""
    return os.environ.get("EMBED_MODEL") or DEFAULT_EMBED_MODEL


class Embedder:
    def __init__(
        self,
        model_name: str = DEFAULT_EMBED_MODEL,
        device: str = "cuda",
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
    ) -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available; pass device='cpu' explicitly to use CPU")
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.model = SentenceTransformer(
            model_name, device=device, model_kwargs={"torch_dtype": dtype}
        )
        self.model.max_seq_length = max_seq_length

    def embed_documents(self, texts: Sequence[str]) -> Any:
        """float32 array (len(texts), dimension), rows L2-normalised."""
        return self._encode(list(texts))

    def embed_query(self, text: str) -> Any:
        """float32 vector (dimension,), L2-normalised."""
        return self._encode([text])[0]

    def _encode(self, texts: list[str]) -> Any:
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype("float32")

    def peak_memory_gb(self) -> float | None:
        """Peak CUDA memory allocated by this process (torch's allocator), or None."""
        import torch

        if not self.device.startswith("cuda"):
            return None
        return torch.cuda.max_memory_allocated(self.device) / 1024**3
