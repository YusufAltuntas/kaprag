"""Page features, rendering and stratified sampling (eval/README.md §7 step 1).

Pages come from the Docling rows of each document. A page is skipped when it is the
cover, a table of contents, near-empty, or mostly page headers/footers. The rest fall
into strata: `table` (table-heavy, numeric), `text` (running text, few tables) or
`mixed` (not sampled). Sampling is per document and stratum with a seeded RNG
(`<seed>:<doc>:<stratum>`), so it does not depend on which other documents are drafted.
"""

from __future__ import annotations

import random
import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from decimal import Decimal
from pathlib import Path

from kaprag.eval.numeric import NumericParseError, parse_number
from kaprag.gold.draft import fold
from kaprag.parsing.outputs import (
    DEFAULT_PARSED_DIR,
    Element,
    iter_elements,
    iter_pages,
    parsed_path,
)

SEED = 42
MAX_PAGE_CHARS = 16_000

MIN_CONTENT_CHARS = 600
TABLE_MIN_CHARS = 800
TABLE_MIN_NUMBERS = 20
TABLE_MIN_SHARE = 0.35
TEXT_MIN_CHARS = 1500
TEXT_MAX_TABLE_SHARE = 0.15
TOC_MIN_LINES = 8

NUMBER_TOKEN = re.compile(r"\(?-?\d[\d.,]*\d\)?|\d")
_TOC_LINE = re.compile(r"\s\d{1,3}$")


@dataclass(frozen=True)
class PageInfo:
    doc_id: str
    page: int
    stratum: str  # table / text / mixed / skip
    skip_reason: str | None
    content_chars: int
    text_chars: int
    table_chars: int
    table_numbers: int
    has_comparative: bool  # a table with current- and prior-year columns

    def to_dict(self) -> dict:
        return asdict(self)


def group_by_page(elements: Iterable[Element]) -> dict[int, list[Element]]:
    pages: dict[int, list[Element]] = defaultdict(list)
    for e in elements:
        pages[e.page].append(e)
    return dict(pages)


def _numbers(text: str) -> int:
    return len(NUMBER_TOKEN.findall(text))


def _has_comparative(table_md: str, year: int) -> bool:
    header = fold(table_md.split("\n", 1)[0])
    current = str(year) in header or "cari dönem" in header
    prior = str(year - 1) in header or "önceki dönem" in header
    return current and prior


def _is_toc(elements: Sequence[Element]) -> bool:
    if any(e.label == "document_index" for e in elements):
        return True
    if any(e.type == "heading" and "içindekiler" in fold(e.text) for e in elements):
        return True
    lines = [e.text.strip() for e in elements if e.type in ("list_item", "paragraph")]
    toc_lines = sum(bool(_TOC_LINE.search(t)) and len(t) < 120 for t in lines)
    return toc_lines >= TOC_MIN_LINES and toc_lines >= 0.5 * len(lines)


def page_info(doc_id: str, year: int, page: int, elements: Sequence[Element]) -> PageInfo:
    """Features and stratum of one page."""
    content = [e for e in elements if e.type != "furniture"]
    furniture = sum(len(e.text) for e in elements if e.type == "furniture")
    content_chars = sum(len(e.text) for e in content)
    text_chars = sum(
        len(e.text) for e in content if e.type in ("paragraph", "list_item") and not e.in_picture
    )
    tables = [e for e in content if e.type == "table"]
    table_chars = sum(len(e.text) for e in tables)
    table_numbers = sum(_numbers(e.text) for e in tables)
    comparative = any(_has_comparative(e.text, year) for e in tables)

    skip = None
    if page == 1:
        skip = "cover"
    elif _is_toc(elements):
        skip = "toc"
    elif content_chars < MIN_CONTENT_CHARS:
        skip = "near_empty"
    elif furniture > content_chars:
        skip = "mostly_furniture"
    if skip:
        stratum = "skip"
    elif (
        table_chars >= TABLE_MIN_CHARS
        and table_numbers >= TABLE_MIN_NUMBERS
        and table_chars >= TABLE_MIN_SHARE * content_chars
    ):
        stratum = "table"
    elif text_chars >= TEXT_MIN_CHARS and table_chars <= TEXT_MAX_TABLE_SHARE * content_chars:
        stratum = "text"
    else:
        stratum = "mixed"
    return PageInfo(
        doc_id=doc_id,
        page=page,
        stratum=stratum,
        skip_reason=skip,
        content_chars=content_chars,
        text_chars=text_chars,
        table_chars=table_chars,
        table_numbers=table_numbers,
        has_comparative=comparative,
    )


def sample_pages(
    infos: Sequence[PageInfo], n_table: int, n_text: int, seed: int = SEED
) -> list[PageInfo]:
    """Seeded sample of `n_table` table pages and `n_text` text pages of ONE document,
    in page order. A stratum with fewer pages contributes all of them."""
    out: list[PageInfo] = []
    doc_ids = {i.doc_id for i in infos}
    if len(doc_ids) > 1:
        raise ValueError(f"sample one document at a time, got {sorted(doc_ids)}")
    for stratum, n in (("table", n_table), ("text", n_text)):
        pool = sorted((i for i in infos if i.stratum == stratum), key=lambda i: i.page)
        if not pool or n <= 0:
            continue
        rng = random.Random(f"{seed}:{pool[0].doc_id}:{stratum}")
        out += rng.sample(pool, min(n, len(pool)))
    return sorted(out, key=lambda i: i.page)


def render_page(elements: Sequence[Element], max_chars: int = MAX_PAGE_CHARS) -> str:
    """Prompt rendering of a page: headings, paragraphs, list items, Markdown tables and
    picture text, in reading order; page headers/footers dropped. The nearest heading
    before the page (from the first element's section path) is given as context."""
    content = [e for e in elements if e.type != "furniture"]
    parts: list[str] = []
    if content and content[0].type != "heading" and content[0].section_path:
        parts.append(f"[Önceki başlık: {content[0].section_path[-1]}]")
    for e in content:
        text = e.text.strip()
        if not text:
            continue
        if e.in_picture:
            parts.append(f"[Görsel içi metin] {text}")
        elif e.type == "heading":
            parts.append(f"## {text}")
        elif e.type == "table":
            caption = f"[Tablo: {e.caption}]\n" if e.caption else "[Tablo]\n"
            parts.append(caption + text)
        elif e.type == "list_item":
            parts.append(f"- {text}")
        else:
            parts.append(text)
    rendered = "\n\n".join(parts)
    if len(rendered) > max_chars:
        rendered = rendered[:max_chars] + "\n[... sayfanın geri kalanı kısaltıldı]"
    return rendered


def numbers_in(text: str) -> set[Decimal]:
    """Every number written in `text` (Turkish format; parentheses read as negative,
    and their absolute value is added too, since a quote may drop the parentheses)."""
    found: set[Decimal] = set()
    for token in NUMBER_TOKEN.findall(text):
        try:
            value = parse_number(token)
        except NumericParseError:
            continue
        found.add(value)
        found.add(abs(value))
    return found


@dataclass
class DocPages:
    """Docling elements and PyMuPDF text of one document, by 1-based page."""

    doc_id: str
    elements: dict[int, list[Element]]
    plain: dict[int, str]

    @classmethod
    def load(cls, doc_id: str, parsed_dir: Path = DEFAULT_PARSED_DIR) -> DocPages:
        elements = group_by_page(iter_elements(parsed_path("docling", doc_id, parsed_dir)))
        plain = {p.page: p.text for p in iter_pages(parsed_path("pymupdf", doc_id, parsed_dir))}
        return cls(doc_id=doc_id, elements=elements, plain=plain)

    def render(self, page: int) -> str:
        return render_page(self.elements.get(page, []))

    def numbers(self, page: int) -> set[Decimal]:
        """Numbers on a page, from both parsers (Docling may split or merge cells)."""
        docling = "\n".join(e.text for e in self.elements.get(page, []))
        return numbers_in(docling) | numbers_in(self.plain.get(page, ""))

    def infos(self, year: int) -> list[PageInfo]:
        return [
            page_info(self.doc_id, year, page, els) for page, els in sorted(self.elements.items())
        ]
