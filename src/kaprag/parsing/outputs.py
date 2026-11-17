"""On-disk formats of the parsed corpus, and readers for them.

Both parsers write one JSON Lines file per document, UTF-8:

`data/parsed/pymupdf/<doc_id>.jsonl` - one row per PDF page, in page order, every
page present (text may be empty):

    {"page": int, "text": str}

`data/parsed/docling/<doc_id>.jsonl` - one row per document element, in Docling's
reading order (see `kaprag.parsing.docling_parser` for how rows are built):

    {
      "page": int,                # first PDF page the element is on
      "pages": [int, ...],        # all PDF pages the element has provenance on, sorted
      "type": str,                # one of ELEMENT_TYPES
      "label": str,               # Docling's own label (e.g. "section_header", "document_index")
      "level": int | null,        # heading level for type "heading" (0 = title), else null
      "section_path": [str, ...], # headings the element sits under, outermost first;
                                  # for a heading: its ancestors, not itself
      "text": str,                # tables rendered as Markdown
      "caption": str | null       # a table's caption, when Docling linked one
    }

`page` is always the 1-based PDF page index, not the printed page number.
"""

import json
import os
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from kaprag.corpus import REPO_ROOT

DEFAULT_PARSED_DIR = REPO_ROOT / "data" / "parsed"
PARSERS = ("pymupdf", "docling")

ELEMENT_TYPES = (
    "heading",
    "paragraph",
    "list_item",
    "table",
    "caption",
    "furniture",  # page headers and footers
    "other",
)


@dataclass(frozen=True)
class PageText:
    """One PyMuPDF row: the plain text of one PDF page."""

    page: int
    text: str


@dataclass(frozen=True)
class Element:
    """One Docling row: a document element in reading order."""

    page: int
    pages: list[int]
    type: str
    label: str
    level: int | None
    section_path: list[str]
    text: str
    caption: str | None


def parsed_path(parser: str, doc_id: str, parsed_dir: Path = DEFAULT_PARSED_DIR) -> Path:
    """Output file of a parser for a document: `<parsed_dir>/<parser>/<doc_id>.jsonl`."""
    if parser not in PARSERS:
        raise ValueError(f"unknown parser {parser!r}, expected one of {PARSERS}")
    return parsed_dir / parser / f"{doc_id}.jsonl"


def iter_jsonl(path: Path) -> Iterator[dict]:
    """Rows of a JSON Lines file."""
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: {e}") from e


def write_jsonl_atomic(path: Path, rows: Iterable[dict]) -> int:
    """Write rows as JSON Lines, replacing `path` atomically; returns the row count.

    The file only appears under its final name once it is complete, so a reader
    (or a resumed run) never sees a half-written file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    count = 0
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)
    return count


def iter_pages(path: Path) -> Iterator[PageText]:
    """PyMuPDF rows of one parsed document file."""
    for row in iter_jsonl(path):
        _check_keys(path, row, PageText)
        yield PageText(**row)


def iter_elements(path: Path) -> Iterator[Element]:
    """Docling rows of one parsed document file."""
    for row in iter_jsonl(path):
        _check_keys(path, row, Element)
        yield Element(**row)


def _check_keys(path: Path, row: dict, cls: type) -> None:
    expected = set(cls.__dataclass_fields__)
    if set(row) != expected:
        raise ValueError(f"{path}: row keys {sorted(row)} != {sorted(expected)}")
