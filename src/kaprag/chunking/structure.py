"""The structure-aware chunker over Docling elements.

Rules (rows in Docling's reading order):

- `furniture` rows (page headers/footers) are dropped, as are empty rows.
- Running text (`paragraph`, `list_item`, `caption`, `other`) is grouped by
  `section_path`: consecutive running-text rows with the same path form one group, and a
  heading or a path change closes it. A group is packed into chunks of at most `size`
  tokens; every chunk starts with the nearest heading (last element of `section_path`)
  as a prefix. A single row longer than the budget is split by tokens. Table captions
  are not separate rows (the parser puts them on the table), so every caption row here
  is running text.
- A `table` is its own chunk (`chunk_type` "table"), prefixed with the nearest heading
  and its caption. A table over the budget is split into row groups, each repeating the
  prefix and the Markdown header rows. All parts keep the table's page span.
- Rows found inside pictures (`in_picture`; KPI boxes, chart labels) stay out of the
  running-text groups: they are collected per page into `picture_text` chunks, prefixed
  with the nearest heading of the page's first such row.

Only the nearest heading is used as text, because Docling's levels (and so the deeper
`section_path`) are approximate; the full path is kept as metadata. Chunk ordinals
follow the reading order of each chunk's first row. Lengths are measured with the
injected tokenizer; a packed chunk can exceed `size` by the separator tokens between
rows (a few tokens); a single table row longer than the budget is split by tokens
(each part still under the repeated header). Header rows taking more than half the
budget (with the prefix) are not repeated but chunked like body rows.
"""

import re
from collections.abc import Iterable
from dataclasses import dataclass, field

from kaprag.chunking.base import (
    DEFAULT_CHUNK_TOKENS,
    Chunk,
    TokenSpans,
    count_tokens,
    split_by_tokens,
)
from kaprag.parsing.outputs import Element

RUNNING_TEXT_TYPES = frozenset({"paragraph", "list_item", "caption", "other"})
PREFIX_SEPARATOR = "\n\n"
ROW_SEPARATOR = "\n"
_TABLE_RULE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$")


@dataclass
class _Piece:
    """A chunk before ordinals are assigned; `order` sorts pieces into reading order."""

    order: tuple[int, int]
    chunk_type: str
    section_path: tuple[str, ...]
    text: str
    page_start: int
    page_end: int


@dataclass
class _Group:
    first_index: int
    section_path: tuple[str, ...]
    rows: list[Element] = field(default_factory=list)


def chunk_structure(
    doc_id: str,
    elements: Iterable[Element],
    token_spans: TokenSpans,
    size: int = DEFAULT_CHUNK_TOKENS,
) -> list[Chunk]:
    pieces: list[_Piece] = []
    pictures: dict[int, _Group] = {}  # page -> picture-text rows on that page
    group: _Group | None = None

    def close_group() -> None:
        nonlocal group
        if group is not None:
            pieces.extend(_pack(group, "text", token_spans, size))
            group = None

    for index, element in enumerate(elements):
        if element.type == "furniture" or not element.text.strip():
            continue
        if element.in_picture:
            pictures.setdefault(
                element.page, _Group(index, tuple(element.section_path))
            ).rows.append(element)
        elif element.type == "heading":
            close_group()
        elif element.type == "table":
            pieces.extend(_table_pieces(index, element, token_spans, size))
        elif element.type in RUNNING_TEXT_TYPES:
            path = tuple(element.section_path)
            if group is not None and group.section_path != path:
                close_group()
            if group is None:
                group = _Group(index, path)
            group.rows.append(element)
        else:
            raise ValueError(f"{doc_id}: unknown element type {element.type!r}")
    close_group()
    for picture_group in pictures.values():
        pieces.extend(_pack(picture_group, "picture_text", token_spans, size))

    pieces.sort(key=lambda p: p.order)
    return [
        Chunk(
            doc_id=doc_id,
            ordinal=ordinal,
            page_start=p.page_start,
            page_end=p.page_end,
            chunk_type=p.chunk_type,
            section_path=p.section_path,
            text=p.text,
            n_tokens=count_tokens(p.text, token_spans),
        )
        for ordinal, p in enumerate(pieces)
    ]


def _heading(section_path: Iterable[str]) -> str:
    path = list(section_path)
    return path[-1].strip() if path else ""


def _with_prefix(prefix: str, body: str) -> str:
    return f"{prefix}{PREFIX_SEPARATOR}{body}" if prefix else body


def _page_span(rows: Iterable[Element]) -> tuple[int, int]:
    pages = [p for row in rows for p in (row.page, *row.pages)]
    return min(pages), max(pages)


def _pack(group: _Group, chunk_type: str, token_spans: TokenSpans, size: int) -> list[_Piece]:
    """Pack a group's rows, whole where possible, into prefixed chunks of <= `size` tokens."""
    prefix = _heading(group.section_path)
    budget = size - (count_tokens(_with_prefix(prefix, ""), token_spans) if prefix else 0)
    # (text, rows it came from) per chunk body
    bodies: list[tuple[list[str], list[Element]]] = []
    texts: list[str] = []
    rows: list[Element] = []
    used = 0

    def flush() -> None:
        nonlocal texts, rows, used
        if texts:
            bodies.append((texts, rows))
        texts, rows, used = [], [], 0

    for row in group.rows:
        text = row.text.strip()
        n = count_tokens(text, token_spans)
        if n > budget:
            flush()
            bodies.extend(([part], [row]) for part in split_by_tokens(text, token_spans, budget))
            continue
        if used + n > budget:
            flush()
        texts.append(text)
        rows.append(row)
        used += n
    flush()

    pieces = []
    for i, (body_texts, body_rows) in enumerate(bodies):
        start, end = _page_span(body_rows)
        pieces.append(
            _Piece(
                order=(group.first_index, i),
                chunk_type=chunk_type,
                section_path=group.section_path,
                text=_with_prefix(prefix, ROW_SEPARATOR.join(body_texts)),
                page_start=start,
                page_end=end,
            )
        )
    return pieces


def _split_table(markdown: str) -> tuple[list[str], list[str]]:
    """(header lines, body lines) of a Markdown table: the header ends at the first
    `|---|` rule; without a rule, the first line is the header."""
    lines = markdown.strip().split("\n")
    for i, line in enumerate(lines):
        if _TABLE_RULE.match(line):
            return lines[: i + 1], lines[i + 1 :]
    return lines[:1], lines[1:]


def _table_pieces(index: int, element: Element, token_spans: TokenSpans, size: int) -> list[_Piece]:
    prefix = "\n".join(
        part for part in (_heading(element.section_path), (element.caption or "").strip()) if part
    )
    path = tuple(element.section_path)
    start, end = _page_span([element])
    table = element.text.strip()

    def piece(i: int, body: str) -> _Piece:
        return _Piece((index, i), "table", path, _with_prefix(prefix, body), start, end)

    if count_tokens(_with_prefix(prefix, table), token_spans) <= size:
        return [piece(0, table)]

    header_lines, body_lines = _split_table(table)
    header_tokens = count_tokens(
        _with_prefix(prefix, ROW_SEPARATOR.join(header_lines)), token_spans
    )
    if header_tokens > size // 2:
        # Degenerate header (e.g. a whole paragraph in a header cell): repeating it would
        # leave almost no room for rows, so it is chunked like the rows instead.
        header_lines, body_lines = [], header_lines + body_lines
    header = [ROW_SEPARATOR.join(header_lines)] if header_lines else []
    budget = size - count_tokens(
        _with_prefix(prefix, ROW_SEPARATOR.join([*header, "x"])), token_spans
    )
    groups: list[list[str]] = []
    current: list[str] = []
    used = 0
    for line in body_lines:
        n = count_tokens(line, token_spans)
        if n > budget:  # a single row over the budget: split it by tokens
            if current:
                groups.append(current)
                current, used = [], 0
            groups.extend([part] for part in split_by_tokens(line, token_spans, budget))
            continue
        if current and used + n > budget:
            groups.append(current)
            current, used = [], 0
        current.append(line)
        used += n
    if current:
        groups.append(current)
    return [piece(i, ROW_SEPARATOR.join([*header, *rows])) for i, rows in enumerate(groups)]
