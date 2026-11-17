"""Prompt context: retrieved chunks as numbered sources `[1]..[k]`.

Each source is a label line (bank, report year, doc id, PDF page span) followed by the
chunk text. The model cites by source number; `kaprag.rag.postprocess` maps numbers
back to pages, so numbering here is the list order (1-based) and nothing else.
"""

from __future__ import annotations

from collections.abc import Sequence

from kaprag.eval.pipeline import RetrievedItem
from kaprag.gold.draft import BANK_NAMES


def _bank_and_year(doc: str) -> tuple[str, str]:
    """`akbank-2024` -> (`Akbank`, `2024`); unknown banks keep their id."""
    bank, _, year = doc.rpartition("-")
    return BANK_NAMES.get(bank, bank), year


def _pages(item: RetrievedItem) -> str:
    if item.page_start == item.page_end:
        return str(item.page_start)
    return f"{item.page_start}-{item.page_end}"


def source_label(number: int, item: RetrievedItem) -> str:
    bank, year = _bank_and_year(item.doc)
    return f"[{number}] Banka: {bank} | Yıl: {year} | Doküman: {item.doc} | Sayfa: {_pages(item)}"


def format_context(items: Sequence[RetrievedItem]) -> str:
    """Sources numbered from 1 in the given order, separated by blank lines."""
    blocks = [
        f"{source_label(n, item)}\n{(item.text or '').strip()}"
        for n, item in enumerate(items, start=1)
    ]
    return "\n\n".join(blocks)
