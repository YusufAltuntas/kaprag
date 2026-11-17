"""The naive baseline chunker: fixed-size token windows over PyMuPDF page text.

The document's non-empty pages are joined in page order with a blank line between them
and cut into windows of `size` tokens, consecutive windows sharing `overlap` tokens.
Windows cross page boundaries; each chunk records the first and last PDF page it has
tokens from. No layout heuristics, no header/footer stripping.
"""

from collections.abc import Iterable

from kaprag.chunking.base import (
    DEFAULT_CHUNK_TOKENS,
    DEFAULT_OVERLAP_TOKENS,
    Chunk,
    TokenSpans,
    token_windows,
)
from kaprag.parsing.outputs import PageText

PAGE_SEPARATOR = "\n\n"


def chunk_fixed(
    doc_id: str,
    pages: Iterable[PageText],
    token_spans: TokenSpans,
    size: int = DEFAULT_CHUNK_TOKENS,
    overlap: int = DEFAULT_OVERLAP_TOKENS,
) -> list[Chunk]:
    parts: list[str] = []
    # Per token: (start_char, end_char, page) in the joined document text.
    tokens: list[tuple[int, int, int]] = []
    offset = 0
    for page in pages:
        spans = token_spans(page.text)
        if not spans:
            continue
        if parts:
            parts.append(PAGE_SEPARATOR)
            offset += len(PAGE_SEPARATOR)
        parts.append(page.text)
        tokens.extend((offset + a, offset + b, page.page) for a, b in spans)
        offset += len(page.text)
    text = "".join(parts)

    return [
        Chunk(
            doc_id=doc_id,
            ordinal=ordinal,
            page_start=tokens[a][2],
            page_end=tokens[b - 1][2],
            chunk_type="text",
            section_path=(),
            text=text[tokens[a][0] : tokens[b - 1][1]],
            n_tokens=b - a,
        )
        for ordinal, (a, b) in enumerate(token_windows(len(tokens), size, overlap))
    ]
