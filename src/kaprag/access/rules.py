"""The indexing-time role rule (SIMULATED access model, see `kaprag.access.roles`).

Deterministic, from the parsed Docling rows of a document; no LLM.

1. Risk-management headings (`heading_roles`): a heading containing "risk yönetimi"
   after Turkish case folding (`fold`: İ, I and ı all become i, so "RİSK YÖNETİMİ",
   "RISK YÖNETIMI" and "Risk yönetimi" are equal; "Risk Yönetimine İlişkin
   Açıklamalar" matches too) -> {risk}. A row is under such a heading when the heading
   is ANY element of its `section_path` (or the row is that heading): Docling's
   heading levels are approximate, so the path may be too deep or too shallow, and
   matching any element errs towards restriction.

2. Statutory report sections: the BDDK financial reports inside the annual reports
   (Akbank, İş Bankası, Yapı Kredi; Garanti publishes them separately) are split into
   seven numbered sections. Docling labels their titles inconsistently (a heading at
   any level, or a plain paragraph: İş Bankası 2023's "BEŞİNCİ BÖLÜM: ..."), so they
   are tracked in reading order instead: a row (any non-furniture type) whose text
   STARTS with a section title ("DÖRDÜNCÜ BÖLÜM", "Beşinci bölüm - ...", optionally
   after the bank's "... A.Ş.") opens that section until the next title
   (`statutory_section`). A table of contents is a run of titles: its page gets the
   restricted roles too, and the real titles after it reset the section. Only
   precedence is guaranteed, not exact boundaries: a page where section 4 ends is
   {risk}, and a running page header that Docling reads as a title opens the section
   early (Yapı Kredi 2024: accounting-policy pages under a "Dördüncü Bölüm" header).

   - section 4, "Mali bünyeye ve risk yönetimine ilişkin bilgiler" (capital, credit,
     market and liquidity risk disclosures; its title contains the risk phrase) -> {risk}
   - section 5, "Finansal tablolara ilişkin açıklama ve dipnotlar": the
     financial-statement footnotes                                  -> {analyst, risk}
   - sections 1-3, 6, 7 (general information, the statements themselves, accounting
     policies, other disclosures, auditor's report) and everything outside a
     statutory report                                               -> no section role

   Footnotes are defined by section 5 only, not by a "dipnot" heading phrase: Akbank
   and Yapı Kredi repeat the report's running title "... FİNANSAL TABLOLARA İLİŞKİN
   AÇIKLAMA VE DİPNOTLAR" as a heading on most pages of all seven sections, so a
   phrase rule would turn their statements, accounting policies and auditor's report
   into footnotes but not İş Bankası's.

A row's role set is the most restrictive of its heading role and its section role
(furniture and empty rows are ignored, as by the chunkers). `DocumentAccess` holds
per PDF page the most restrictive role set of the rows on it (`page_roles`) and of
their sections alone (`section_roles`). A chunk's role set (`chunk_roles`):

- `fixed` (no headings): the most restrictive `page_roles` over its page span, so a
  chunk mixing public and restricted pages is restricted.
- `structure`: the heading role of its own `section_path` combined with the
  `section_roles` of its pages; a `picture_text` chunk (whose path is only its page's
  first picture row's) uses `page_roles` like a fixed chunk.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from kaprag.access.roles import FOOTNOTES, PUBLIC, RISK, most_restrictive
from kaprag.parsing.outputs import Element

RISK_PHRASE = "risk yönetimi"

_ORDINALS = {
    "birinci": 1,
    "ikinci": 2,
    "üçüncü": 3,
    "dördüncü": 4,
    "beşinci": 5,
    "altinci": 6,  # "altıncı" after folding
    "yedinci": 7,
}
# A section title starts the row (optionally after the bank's legal name, "... a.ş."):
# "DÖRDÜNCÜ BÖLÜM: MALİ BÜNYE...", "Beşinci bölüm - ...", "Yapı ve Kredi Bankası A.Ş.
# Üçüncü Bölüm". Cross-references inside running text ("Dördüncü Bölüm II no.lu
# dipnotta", "Bu konuda Dördüncü Bölüm'de") do not start a row, or are followed by a
# note number or a suffix, and are not titles. A Turkish case or plural suffix glued to
# "bölüm" ("Birinci bölümde", "Beşinci bölümün"; -ü, -e, -de/-den, -le/-ler) marks
# running text too. No plain word boundary: a real title can be parsed with a stray
# letter ("BEŞİNCİ BÖLÜMs", akbank-2024 p.152).
_SECTION = re.compile(
    r"^(?:[^.:\n]{0,60}a\.ş\.\s*)?(" + "|".join(_ORDINALS) + r") bölüm"
    r"(?!ü|e|de|le)(?!['’])(?!\s*,)(?!\s*[ivx0-9]+(?:[.\-]\w{1,4})*\s*n(?:o|um))"
)
SECTION_ROLES = {4: RISK, 5: FOOTNOTES}

_DOTTED_I = str.maketrans({"İ": "i", "I": "i", "ı": "i"})


def fold(text: str) -> str:
    """Case- and dotless-i-insensitive form of Turkish text, whitespace collapsed."""
    text = unicodedata.normalize("NFC", text).translate(_DOTTED_I).lower()
    return " ".join(text.split())


def heading_roles(heading: str) -> frozenset[str]:
    """{risk} for a risk-management heading, else {public} (rule 1)."""
    return RISK if RISK_PHRASE in fold(heading) else PUBLIC


def path_roles(section_path: Iterable[str]) -> frozenset[str]:
    """The most restrictive heading role along a section path."""
    return most_restrictive(heading_roles(h) for h in section_path)


def statutory_section(text: str) -> int | None:
    """The number of the statutory report section whose title starts `text`
    ("DÖRDÜNCÜ BÖLÜM: ..." -> 4), else None (rule 2)."""
    match = _SECTION.match(fold(text))
    return _ORDINALS[match.group(1)] if match else None


@dataclass(frozen=True)
class DocumentAccess:
    """Per-page role sets of one document (pages without rows are absent)."""

    page_roles: Mapping[int, frozenset[str]] = field(default_factory=dict)
    section_roles: Mapping[int, frozenset[str]] = field(default_factory=dict)

    @classmethod
    def from_elements(cls, elements: Iterable[Element]) -> DocumentAccess:
        pages: dict[int, frozenset[str]] = {}
        sections: dict[int, frozenset[str]] = {}
        section = PUBLIC
        for e in elements:
            if e.type == "furniture" or not e.text.strip():
                continue
            path = list(e.section_path)
            if e.type == "heading":
                path.append(e.text)
            number = statutory_section(e.text)
            if number is not None:
                section = SECTION_ROLES.get(number, PUBLIC)
            roles = most_restrictive([path_roles(path), section])
            for page in {e.page, *e.pages}:
                pages[page] = most_restrictive([pages.get(page, PUBLIC), roles])
                sections[page] = most_restrictive([sections.get(page, PUBLIC), section])
        return cls(pages, sections)

    def pages_roles(self, page_start: int, page_end: int) -> frozenset[str]:
        """Most restrictive `page_roles` over a page span (inclusive)."""
        return _span(self.page_roles, page_start, page_end)

    def pages_section_roles(self, page_start: int, page_end: int) -> frozenset[str]:
        """Most restrictive `section_roles` over a page span (inclusive)."""
        return _span(self.section_roles, page_start, page_end)


def _span(by_page: Mapping[int, frozenset[str]], start: int, end: int) -> frozenset[str]:
    return most_restrictive(by_page.get(p, PUBLIC) for p in range(start, end + 1))


def chunk_roles(
    variant: str,
    section_path: Sequence[str],
    chunk_type: str,
    page_start: int,
    page_end: int,
    access: DocumentAccess,
) -> frozenset[str]:
    """The role set of a chunk of `variant` (see the module docstring)."""
    if variant == "fixed" or (variant == "structure" and chunk_type == "picture_text"):
        return access.pages_roles(page_start, page_end)
    if variant == "structure":
        return most_restrictive(
            [path_roles(section_path), access.pages_section_roles(page_start, page_end)]
        )
    raise ValueError(f"unknown variant {variant!r}")
