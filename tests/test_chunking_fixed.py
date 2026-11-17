import re

import pytest

from kaprag.chunking.base import split_by_tokens, token_windows
from kaprag.chunking.fixed import chunk_fixed
from kaprag.parsing.outputs import PageText


def ws_spans(text: str) -> list[tuple[int, int]]:
    """Fake tokenizer: one token per whitespace-separated word."""
    return [m.span() for m in re.finditer(r"\S+", text)]


def words(prefix: str, n: int) -> str:
    return " ".join(f"{prefix}{i}" for i in range(n))


def test_token_windows_cover_everything_with_overlap():
    assert token_windows(25, size=10, overlap=2) == [(0, 10), (8, 18), (16, 25)]
    assert token_windows(10, size=10, overlap=2) == [(0, 10)]
    assert token_windows(0, size=10, overlap=2) == []


def test_token_windows_reject_bad_overlap():
    with pytest.raises(ValueError):
        token_windows(5, size=10, overlap=10)


def test_split_by_tokens_keeps_all_words():
    text = words("w", 23)
    parts = split_by_tokens(text, ws_spans, max_tokens=10)
    assert [len(ws_spans(p)) for p in parts] == [10, 10, 3]
    assert " ".join(parts) == text


def test_fixed_chunks_have_size_and_overlap():
    pages = [PageText(1, words("a", 25))]
    chunks = chunk_fixed("d-2023", pages, ws_spans, size=10, overlap=2)
    assert [c.n_tokens for c in chunks] == [10, 10, 9]
    first, second = (c.text.split() for c in chunks[:2])
    assert first[-2:] == second[:2]
    assert chunks[0].text.split()[0] == "a0"
    assert chunks[-1].text.split()[-1] == "a24"
    assert [c.ordinal for c in chunks] == [0, 1, 2]
    assert all(c.chunk_type == "text" and c.section_path == () for c in chunks)
    assert all(c.doc_id == "d-2023" for c in chunks)


def test_fixed_chunks_cross_pages_and_track_them():
    pages = [
        PageText(1, words("a", 6)),
        PageText(2, ""),  # empty page contributes nothing
        PageText(3, words("c", 6)),
        PageText(4, words("d", 6)),
    ]
    chunks = chunk_fixed("d-2023", pages, ws_spans, size=10, overlap=2)
    # tokens: a0-a5 (p1), c0-c5 (p3), d0-d5 (p4) -> windows [0,10) [8,18)
    assert [(c.page_start, c.page_end) for c in chunks] == [(1, 3), (3, 4)]
    assert chunks[0].text.split() == [*words("a", 6).split(), "c0", "c1", "c2", "c3"]
    assert chunks[1].text.split()[:2] == ["c2", "c3"]


def test_fixed_chunk_text_is_a_contiguous_slice_of_the_document():
    pages = [PageText(1, "x  y\nz"), PageText(2, "q r")]
    (chunk,) = chunk_fixed("d-2023", pages, ws_spans, size=10, overlap=2)
    assert chunk.text == "x  y\nz\n\nq r"
    assert (chunk.page_start, chunk.page_end) == (1, 2)


def test_fixed_empty_document_has_no_chunks():
    assert chunk_fixed("d-2023", [PageText(1, "  ")], ws_spans, size=10, overlap=2) == []
