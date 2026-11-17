"""Parsing the annual-report PDFs: a PyMuPDF baseline and a Docling structure-aware parser.

Output formats and their readers live in `kaprag.parsing.outputs`.
"""

from kaprag.parsing.outputs import (
    DEFAULT_PARSED_DIR,
    Element,
    PageText,
    iter_elements,
    iter_pages,
    parsed_path,
)

__all__ = [
    "DEFAULT_PARSED_DIR",
    "Element",
    "PageText",
    "iter_elements",
    "iter_pages",
    "parsed_path",
]
