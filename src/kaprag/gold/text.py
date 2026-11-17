"""Code-side checks on question wording: bank / year / basis mentions and copied phrasing.

A gold question must be answerable without knowing which page it came from (bank,
year and - for financial figures - basis named), and must not copy the page's
wording (eval/README.md §7 step 2), or retrieval looks better than it is.
"""

from __future__ import annotations

import re

from kaprag.gold.draft import fold

BANK_ALIASES: dict[str, tuple[str, ...]] = {
    "garanti": ("garanti",),
    "akbank": ("akbank",),
    "isbank": ("iş bankası", "işbank", "iş bank", "isbank"),
    "yapikredi": ("yapı kredi", "yapıkredi", "yapı ve kredi", "yapikredi"),
}
COPY_NGRAM = 6
_WORD = re.compile(r"\w+")
_YEAR = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")


def mentions_bank(text: str, bank: str) -> bool:
    folded = fold(text)
    return any(alias in folded for alias in BANK_ALIASES[bank])


def banks_mentioned(text: str) -> set[str]:
    return {b for b in BANK_ALIASES if mentions_bank(text, b)}


def years_mentioned(text: str) -> set[int]:
    return {int(y) for y in _YEAR.findall(text)}


def bases_mentioned(text: str) -> set[str]:
    """{'konsolide', 'solo'} words in a text ('konsolide olmayan' counts as solo)."""
    folded = fold(text)
    found: set[str] = set()
    solo_phrases = ("konsolide olmayan", "konsolide edilmemiş", "solo")
    if any(p in folded for p in solo_phrases):
        found.add("solo")
    stripped = folded
    for p in solo_phrases:
        stripped = stripped.replace(p, " ")
    if "konsolide" in stripped:
        found.add("konsolide")
    return found


def copied_run(question: str, page_text: str, n: int = COPY_NGRAM) -> str | None:
    """The first run of `n` consecutive words the question shares with the page, if any."""
    q = _WORD.findall(fold(question))
    p = _WORD.findall(fold(page_text))
    page_grams = {tuple(p[i : i + n]) for i in range(len(p) - n + 1)}
    for i in range(len(q) - n + 1):
        gram = tuple(q[i : i + n])
        if gram in page_grams:
            return " ".join(gram)
    return None
