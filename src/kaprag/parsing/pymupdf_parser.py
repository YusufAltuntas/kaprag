"""Naive baseline parser: PyMuPDF's plain text of each PDF page.

No layout heuristics beyond PyMuPDF's own `page.get_text("text")`, which emits
text blocks in the order they appear in the PDF content stream. On the 2-up spread
pages of this corpus that order is the producer's, not the reader's: blocks of the
left and right printed pages are usually interleaved, and the header band and
navigation strips land wherever they were drawn.
"""

from pathlib import Path

import pymupdf

from kaprag.parsing.outputs import PageText, write_jsonl_atomic


def extract_pages(pdf_path: Path) -> list[PageText]:
    """Plain text of every page, 1-based PDF page index, in page order."""
    with pymupdf.open(pdf_path) as pdf:
        return [
            PageText(page=index, text=page.get_text("text"))
            for index, page in enumerate(pdf, start=1)
        ]


def parse_document(pdf_path: Path, out_path: Path) -> int:
    """Write `out_path` (one row per page) atomically; returns the page count."""
    pages = extract_pages(pdf_path)
    return write_jsonl_atomic(out_path, ({"page": p.page, "text": p.text} for p in pages))
