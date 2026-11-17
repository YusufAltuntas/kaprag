"""Chunk model and token helpers shared by the chunkers.

Chunk sizes are measured in tokens of the embedding model's tokenizer. The tokenizer is
injected as a `TokenSpans` callable (text -> character spans of its tokens, no special
tokens), so the chunkers run without the `ml` group in tests; `load_token_spans` builds
the real one.
"""

from collections.abc import Callable
from dataclasses import dataclass

# text -> [(start_char, end_char), ...] of its tokens, in order, without special tokens.
TokenSpans = Callable[[str], list[tuple[int, int]]]

DEFAULT_CHUNK_TOKENS = 500
DEFAULT_OVERLAP_TOKENS = 50

CHUNK_TYPES = (
    "text",  # running text (all `fixed` chunks; paragraphs/list items under a heading)
    "table",  # a Docling table, whole or a row group of it
    "picture_text",  # text Docling found inside pictures on one page (KPI boxes, charts)
)


@dataclass(frozen=True)
class Chunk:
    """One chunk of a document. Pages are 1-based PDF page indices, inclusive.

    `ordinal` is the chunk's position within its document and variant (0-based, in
    reading order); with `doc_id` and the variant it identifies the chunk.
    `n_tokens` is the length of `text` in tokens, for statistics; it is not stored.
    """

    doc_id: str
    ordinal: int
    page_start: int
    page_end: int
    chunk_type: str
    section_path: tuple[str, ...]
    text: str
    n_tokens: int


def count_tokens(text: str, token_spans: TokenSpans) -> int:
    return len(token_spans(text))


def token_windows(n_tokens: int, size: int, overlap: int) -> list[tuple[int, int]]:
    """Half-open token index windows of `size` covering `n_tokens`, consecutive windows
    sharing `overlap` tokens; the last window ends at `n_tokens`."""
    if not 0 <= overlap < size:
        raise ValueError(f"need 0 <= overlap < size, got overlap={overlap}, size={size}")
    windows = []
    start = 0
    while start < n_tokens:
        end = min(start + size, n_tokens)
        windows.append((start, end))
        if end == n_tokens:
            break
        start = end - overlap
    return windows


def split_by_tokens(text: str, token_spans: TokenSpans, max_tokens: int) -> list[str]:
    """Split `text` into consecutive pieces of at most `max_tokens` tokens (no overlap)."""
    spans = token_spans(text)
    return [
        text[spans[a][0] : spans[b - 1][1]]
        for a, b in token_windows(len(spans), max(max_tokens, 1), 0)
    ]


def load_token_spans(model_name: str) -> TokenSpans:
    """`TokenSpans` of a Hugging Face model's fast tokenizer (needs the `ml` group).

    Uses the Rust backend directly, without truncation, so long pages are measured in
    full and no sequence-length warnings are printed.
    """
    from transformers import AutoTokenizer  # heavy; only with the `ml` group

    backend = AutoTokenizer.from_pretrained(model_name).backend_tokenizer
    backend.no_truncation()
    backend.no_padding()

    def token_spans(text: str) -> list[tuple[int, int]]:
        return list(backend.encode(text, add_special_tokens=False).offsets)

    return token_spans
