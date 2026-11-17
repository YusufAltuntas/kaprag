"""Evidence completion (eval/README.md §3: "Bilgi birden çok sayfada geçiyorsa hepsi
yazılır"): find the other pages of the same report that state a question's answer.

For each answerable single (behaviour `answer`, no `derived_from`):
1. Candidates from the SAME document, original page excluded, at most `MAX_CANDIDATES`:
   - figure scan (numeric questions): pages whose printed numbers equal the expected value
     within the question's tolerance, also in rounded/scaled form ("111 milyar" for
     111.262.000 bin TL; a scale word must be written on the page). A page-YoY
     `hesaplama` needs both of its figures, exactly, on the page;
   - search: dense + full-text (RRF) for the question, restricted to the document.
2. One LLM call per question decides, page by page, whether the page states the answer
   to THIS question (same bank, year, metric, basis; value within tolerance). Only
   passing pages are appended; the original page stays first. A failed call adds nothing.

Derived questions take the union of their sources' completed evidence (sources first,
in `derived_from` order); `answer_with_assumption` belirsiz drafts take their source's.

Bias: candidates come partly from the same retrievers the ablations use (pooling), so a
page no retriever surfaces and that does not print the figure stays unknown.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal

from pydantic import BaseModel

from kaprag.eval.numeric import NumericParseError, parse_number
from kaprag.eval.schema import Evidence
from kaprag.gold.draft import BANK_NAMES, Draft, fold
from kaprag.gold.figures import FigureError, figure_amount
from kaprag.gold.pages import NUMBER_TOKEN, DocPages
from kaprag.retrieval import SearchHit

MAX_CANDIDATES = 6
MAX_FIGURE_CANDIDATES = 4
MIN_COUNT_FOR_SCAN = 100  # small counts (e.g. "7") occur on every page
SEARCH_K = 8
EXCERPT_BEFORE, EXCERPT_AFTER = 1200, 500
SEARCH_EXCERPT = 2500
HEADER_CHARS = 300

SCALE_WORDS: dict[int, tuple[str, ...]] = {
    1: (),
    10**3: ("bin",),
    10**6: ("milyon", "mn", "mio"),
    10**9: ("milyar", "mlr"),
    10**12: ("trilyon", "trn"),
}
_YOY_QUOTE = re.compile(r"^(\d{4}): (.+?); (\d{4}): (.+?) \(")


class PageVerdict(BaseModel):
    candidate: int
    states_answer: bool
    reason: str


class EvidenceCheck(BaseModel):
    items: list[PageVerdict]


@dataclass(frozen=True)
class Candidate:
    page: int
    source: str  # "figure" or "search"
    excerpt: str


SYSTEM = (
    "Sen bir değerlendirme setinin kanıt sayfalarını denetleyen titiz bir finansal "
    "analistsin. Bir sayfayı yalnızca soruyu kendi başına cevaplayan bilgiyi açıkça "
    "içeriyorsa onaylarsın; şüphede kalırsan onaylamazsın."
)


def needs_completion(d: Draft) -> bool:
    """Answerable singles; derived and assumption drafts inherit from their sources."""
    return d.expected_behavior == "answer" and not d.derived_from and bool(d.evidence)


# --- figure scan -------------------------------------------------------------------------


def _number_hits(text: str) -> list[tuple[Decimal, int]]:
    out = []
    for m in NUMBER_TOKEN.finditer(text):
        try:
            out.append((abs(parse_number(m.group(0))), m.start()))
        except NumericParseError:
            continue
    return out


def _within(value: Decimal, target: Decimal, kind: str, amount: Decimal) -> Decimal | None:
    """Relative closeness when `value` matches `target` within tolerance, else None."""
    diff = abs(value - target)
    limit = amount * abs(target) if kind == "rel" else amount
    if diff > limit:
        return None
    return diff / abs(target) if target else diff


@dataclass(frozen=True)
class ScannedText:
    text: str
    numbers: list[tuple[Decimal, int]]  # (absolute value, position)
    words: frozenset[str]


class ScanIndex:
    """Parsed numbers and words of every page text of one document (built once)."""

    def __init__(self, pages: DocPages) -> None:
        self.pages: dict[int, list[ScannedText]] = {}
        for page in sorted(set(pages.elements) | set(pages.plain)):
            texts = [pages.render(page), pages.plain.get(page, "")]
            self.pages[page] = [
                ScannedText(t, _number_hits(t), frozenset(re.findall(r"\w+", fold(t))))
                for t in texts
            ]


def _match_positions(d: Draft, scanned: ScannedText) -> tuple[Decimal, list[int]] | None:
    """(best closeness, positions) of the question's figure(s) in a page text, or None."""
    words, hits = scanned.words, scanned.numbers
    if d.type == "hesaplama":
        m = _YOY_QUOTE.match(d.quote or "")
        if m is None:
            return None
        positions = []
        for figure in (m.group(2), m.group(4)):
            try:
                target = abs(figure_amount(figure)[0])
            except FigureError:
                return None
            found = [pos for value, pos in hits if value == target]
            if not found:
                return None
            positions.append(found[0])
        return Decimal(0), positions
    exp = d.expected
    if exp.value is None or exp.tolerance is None:
        return None
    target = abs(Decimal(str(exp.value)))
    kind, amount = exp.tolerance.type, Decimal(str(exp.tolerance.amount))
    if exp.unit == "adet" and target < MIN_COUNT_FOR_SCAN:
        return None
    scales = list(SCALE_WORDS) if exp.unit == "TL" else [1]
    best: Decimal | None = None
    positions: list[int] = []
    for value, pos in hits:
        if value < 10:  # one-digit numbers match too much at any scale
            continue
        for scale in scales:
            if scale > 1 and not words & set(SCALE_WORDS[scale]):
                continue
            closeness = _within(value * scale, target, kind, amount)
            if closeness is not None:
                if best is None or closeness < best:
                    best = closeness
                positions.append(pos)
                break
    return (best, positions) if best is not None else None


def _excerpt(text: str, positions: Sequence[int]) -> str:
    header = text[:HEADER_CHARS]
    parts = []
    for pos in sorted(positions)[:2]:
        start = max(0, pos - EXCERPT_BEFORE)
        parts.append(text[start : pos + EXCERPT_AFTER])
    body = "\n[...]\n".join(parts)
    if body.startswith(header):
        return body
    return f"{header}\n[...]\n{body}"


def figure_candidates(
    d: Draft, index: ScanIndex, exclude: set[int], limit: int = MAX_FIGURE_CANDIDATES
) -> list[Candidate]:
    """Pages of the draft's document printing its figure(s), closest match first."""
    scored: list[tuple[Decimal, int, Candidate]] = []
    for page, texts in index.pages.items():
        if page in exclude:
            continue
        for scanned in texts:
            match = _match_positions(d, scanned)
            if match is not None:
                closeness, positions = match
                excerpt = _excerpt(scanned.text, positions)
                scored.append((closeness, page, Candidate(page, "figure", excerpt)))
                break
    scored.sort(key=lambda x: (x[0], x[1]))
    return [c for _, _, c in scored[:limit]]


def search_candidates(
    hits: Sequence[SearchHit], doc_id: str, exclude: set[int], limit: int
) -> list[Candidate]:
    """Pages of fused search hits in the draft's document, best first."""
    out: list[Candidate] = []
    seen = set(exclude)
    for h in hits:
        if h.doc_id != doc_id:
            continue
        for page in range(h.page_start, h.page_end + 1):
            if page in seen or len(out) >= limit:
                continue
            seen.add(page)
            out.append(Candidate(page, "search", h.text[:SEARCH_EXCERPT]))
    return out


def candidates_for(
    d: Draft, index: ScanIndex, hits: Sequence[SearchHit], cap: int = MAX_CANDIDATES
) -> list[Candidate]:
    """Figure-scan candidates first (numeric drafts), then search candidates, capped."""
    doc_id = d.evidence[0].doc
    exclude = {e.page for e in d.evidence if e.doc == doc_id and e.page is not None}
    numeric = d.expected.value is not None or d.type == "hesaplama"
    figures = (
        figure_candidates(d, index, exclude, min(cap, MAX_FIGURE_CANDIDATES)) if numeric else []
    )
    taken = exclude | {c.page for c in figures}
    return figures + search_candidates(hits, doc_id, taken, cap - len(figures))


# --- check -------------------------------------------------------------------------------


def build_prompt(d: Draft, candidates: Sequence[Candidate]) -> str:
    bank = ", ".join(BANK_NAMES[b] for b in d.banks)
    lines = [
        f"Soru: {d.question}",
        f"Banka: {bank} | Yıl(lar): {d.years} | Esas: {d.basis or 'belirtilmemiş'}",
        f"Beklenen cevap: {d.expected.answer}",
    ]
    if d.expected.value is not None:
        lines.append(f"Beklenen değer: {d.expected.value} {d.expected.unit}")
    if d.type == "hesaplama":
        lines.append(f"Hesapta kullanılan rakamlar: {d.quote}")
    if d.key_points:
        lines.append("Key points: " + " | ".join(d.key_points))
    lines += [
        "",
        "Aşağıdaki aday sayfaların her biri için karar ver: sayfa, bu sorunun cevabını açıkça "
        "veriyor mu? Aynı banka, aynı yıl, aynı kalem/metrik ve (belirtilmişse) aynı esas "
        "(konsolide/solo) olmalı; rakam yuvarlanmış olabilir (ör. '111 milyar TL') ama "
        "beklenen değerle uyumlu olmalı. Önceki yıl sütunu, başka bir kalem, alt kırılım ya "
        "da başka esas kabul edilmez. Hesaplama sorusunda sayfa iki yılın rakamını da "
        "göstermeli. Sayısal olmayan sorularda sayfa beklenen cevabı ya da en az bir key "
        "point'i açıkça desteklemeli; yalnızca konuyla ilgili olması yetmez. items listesinde "
        "her aday için candidate (aday numarası), states_answer ve kısa reason döndür.",
        "",
    ]
    for i, c in enumerate(candidates):
        lines.append(f"### Aday {i} (sayfa {c.page})\n{c.excerpt}\n")
    return "\n".join(lines)


def apply_verdicts(
    d: Draft, candidates: Sequence[Candidate], check: EvidenceCheck | None
) -> tuple[Draft, dict]:
    """The draft with passing pages appended (original evidence first), and a log record."""
    verdicts = {v.candidate: v for v in check.items} if check is not None else {}
    doc_id = d.evidence[0].doc
    added, rejected = [], []
    for i, c in enumerate(candidates):
        v = verdicts.get(i)
        entry = {"page": c.page, "source": c.source}
        if v is not None and v.states_answer:
            added.append(entry | {"reason": v.reason})
        else:
            reason = "check call failed" if check is None else (v.reason if v else "no verdict")
            rejected.append(entry | {"reason": reason})
    evidence = list(d.evidence) + [Evidence(doc=doc_id, page=a["page"]) for a in added]
    record = {
        "key": d.key,
        "doc": doc_id,
        "original": [asdict_evidence(e) for e in d.evidence],
        "candidates": [{"page": c.page, "source": c.source} for c in candidates],
        "added": added,
        "rejected": rejected,
    }
    return d.model_copy(update={"evidence": evidence}), record


def asdict_evidence(e: Evidence) -> dict:
    return {"doc": e.doc, "page": e.page}


def propagate(drafts: Sequence[Draft]) -> list[Draft]:
    """Derived / assumption drafts get the union of their sources' (completed) evidence,
    sources in `derived_from` order, after their own evidence."""
    by_key = {d.key: d for d in drafts}
    out = []
    for d in drafts:
        if d.derived_from and d.expected_behavior in ("answer", "answer_with_assumption"):
            seen: dict[tuple[str, int | None], Evidence] = {}
            for e in d.evidence:
                seen.setdefault((e.doc, e.page), e)
            for k in d.derived_from:
                src = by_key.get(k)
                for e in src.evidence if src is not None else []:
                    seen.setdefault((e.doc, e.page), e)
            d = d.model_copy(update={"evidence": list(seen.values())})
        out.append(d)
    return out
