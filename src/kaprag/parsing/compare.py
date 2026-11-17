"""Side-by-side view of what the two parsers produced for one PDF page.

Works only from parsed outputs on disk (`kaprag.parsing.outputs` formats).
"""

import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from kaprag.corpus import Document
from kaprag.parsing.outputs import Element, PageText, iter_elements, iter_pages

# Digit groups with optional thousand/decimal separators: "12.778", "1.592.375.704", "12,5".
NUMERIC_TOKEN = re.compile(r"\d+(?:[.,]\d+)*")


def numeric_tokens(text: str) -> list[str]:
    return NUMERIC_TOKEN.findall(text)


def page_elements(elements: Iterable[Element], page: int) -> list[Element]:
    """Elements with provenance on `page`, in reading order."""
    return [e for e in elements if page in e.pages]


def pick_table_page(elements: list[Element]) -> int | None:
    """The page with the most Docling tables (ties: most table text, then lowest page)."""
    per_page: dict[int, tuple[int, int]] = {}
    for e in elements:
        if e.type == "table":
            count, chars = per_page.get(e.page, (0, 0))
            per_page[e.page] = (count + 1, chars + len(e.text))
    if not per_page:
        return None
    return max(per_page, key=lambda p: (*per_page[p], -p))


def default_choices(docs: list[Document], docling_dir: Path) -> list[tuple[str, int]]:
    """For each bank, the first manifest document with Docling tables, and its table page."""
    choices: list[tuple[str, int]] = []
    seen_banks: set[str] = set()
    for doc in docs:
        path = docling_dir / f"{doc.doc_id}.jsonl"
        if doc.bank in seen_banks or not path.is_file():
            continue
        page = pick_table_page(list(iter_elements(path)))
        if page is not None:
            choices.append((doc.doc_id, page))
            seen_banks.add(doc.bank)
    return choices


def render_comparison(
    doc_id: str, page: int, pymupdf_page: PageText, elements: list[Element]
) -> str:
    """Facts, then the PyMuPDF text, then the Docling elements, for one page."""
    docling_text = "\n".join(e.text for e in elements)
    mu_numbers = Counter(numeric_tokens(pymupdf_page.text))
    dl_numbers = Counter(numeric_tokens(docling_text))
    tables = [e for e in elements if e.type == "table"]
    types = Counter(e.type for e in elements)
    mu_count, dl_count = sum(mu_numbers.values()), sum(dl_numbers.values())
    lines = [
        "=" * 100,
        f"{doc_id}  PDF page {page}",
        "=" * 100,
        f"PyMuPDF : {len(pymupdf_page.text):>6} chars, {mu_count:>5} numeric tokens",
        f"Docling : {len(docling_text):>6} chars, {dl_count:>5} numeric tokens, "
        f"{len(elements)} elements, {len(tables)} tables",
        "Docling element types: " + ", ".join(f"{t}={n}" for t, n in sorted(types.items())),
        f"numeric tokens in PyMuPDF but not in Docling: {sum((mu_numbers - dl_numbers).values())}; "
        f"in Docling but not in PyMuPDF: {sum((dl_numbers - mu_numbers).values())}",
        "",
        "-" * 40 + " PyMuPDF " + "-" * 51,
        pymupdf_page.text.rstrip(),
        "",
        "-" * 40 + " Docling " + "-" * 51,
    ]
    for e in elements:
        head = f"[{e.type}"
        if e.level is not None:
            head += f" L{e.level}"
        head += f" p{','.join(map(str, e.pages))}]"
        if e.in_picture:
            head = head[:-1] + " in-picture]"
        if e.section_path:
            head += " " + " > ".join(e.section_path)
        lines.append(head)
        if e.caption:
            lines.append(f"(caption) {e.caption}")
        lines.append(e.text.rstrip())
        lines.append("")
    return "\n".join(lines)


def compare_page(doc_id: str, page: int, pymupdf_dir: Path, docling_dir: Path) -> str:
    """Render the comparison for one page from the two parsed files of a document."""
    pymupdf_pages = {p.page: p for p in iter_pages(pymupdf_dir / f"{doc_id}.jsonl")}
    if page not in pymupdf_pages:
        raise ValueError(f"{doc_id}: page {page} not in PyMuPDF output")
    elements = page_elements(iter_elements(docling_dir / f"{doc_id}.jsonl"), page)
    return render_comparison(doc_id, page, pymupdf_pages[page], elements)
