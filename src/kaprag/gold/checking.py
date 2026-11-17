"""Step 3: a second Sonnet call per page checks all of that page's drafts against it.

A draft passes only with an explicit verdict that the answer (every key point, the
figure, its unit and year column, the stated basis) is on the page and the question
is self-contained. Passing drafts get `llm_checked=True`; the rest are dropped with the
checker's reason. A draft without a verdict (call failed, index missing) is dropped
too: it is never marked checked.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel

from kaprag.gold.draft import BANK_NAMES, Draft
from kaprag.gold.metrics import METRICS_BY_ID


class CheckItem(BaseModel):
    index: int
    supported: bool
    figure_ok: bool | None
    basis_ok: bool
    self_contained: bool
    reason: str


class PageCheck(BaseModel):
    items: list[CheckItem]


SYSTEM = (
    "Sen bir finansal değerlendirme setinin titiz denetçisisin. Taslak soru-cevapları, "
    "kaynak sayfayla karşılaştırıp yalnızca sayfada açıkça dayanağı olanları onaylarsın. "
    "Şüphede kalırsan onaylamazsın."
)

_INSTRUCTIONS = """Her taslak için şunları değerlendir ve items listesinde index ile döndür:
- supported: Cevap (sentezde her bir key point) sayfada açıkça yazıyor mu? Dış bilgi sayılmaz.
- figure_ok: Sayısal taslaklarda: rakam sayfadaki doğru kalemden ve doğru yıl sütunundan mı \
alınmış, unit_text sayfada o rakama uygulanan birim mi (ör. tablo "bin TL" ise "bin TL"), \
normalize değer doğru mu? Hesaplama taslaklarında iki yılın rakamı da doğru mu? Sayısal \
olmayanlarda null.
- basis_ok: Taslağın esası (konsolide/solo) sayfanın belirttiğiyle uyumlu mu? Taslak esası null \
ise ve sayfa da açıkça belirtmiyorsa true; sayfa belirtiyorsa ama taslak null ise ya da \
yanlışsa false.
- self_contained: Soru banka adını ve yılı içeriyor ve hangi sayfadan geldiği bilinmeden \
anlaşılır mı; soru ile cevap birbiriyle tutarlı mı?
- reason: Kısa gerekçe (özellikle bir alan false ise neden)."""


def _describe(index: int, d: Draft, multi_page: bool) -> str:
    head = f"[{index}] tip={d.type} yıl(lar)={d.years} esas={d.basis}"
    if multi_page:
        head += f" kanıt sayfası={d.evidence[0].page}"
    lines = [head, f"Soru: {d.question}"]
    if d.source == "probe" and d.metric in METRICS_BY_ID:
        metric = METRICS_BY_ID[d.metric]
        lines.append(f"İstenen metrik: {metric.description}")
        if not metric.needs_basis:
            lines.append(
                "Not: Adet metriği; esas (konsolide/solo) kullanılmaz. basis_ok'u, rakamın "
                "Banka'nın kendisine (Grup değil) ait olup olmadığına göre değerlendir."
            )
    lines.append(f"Cevap: {d.expected.answer}")
    if d.type == "hesaplama":
        lines.append(f"Kullanılan rakamlar (yıl: rakam, birim): {d.quote}")
    elif d.expected.value is not None:
        lines.append(
            f"Rakam (sayfada yazdığı gibi): {d.quote} | normalize: {d.expected.value} "
            f"{d.expected.unit}"
        )
    if d.key_points:
        lines.append("Key points: " + " | ".join(d.key_points))
    return "\n".join(lines)


def build_prompt(
    bank: str, year: int, pages: Sequence[tuple[int, str]], drafts: Sequence[Draft]
) -> str:
    """Check prompt for drafts of one report; `pages` are (PDF page, rendered text)."""
    multi = len(pages) > 1
    listing = "\n\n".join(_describe(i, d, multi) for i, d in enumerate(drafts))
    if multi:
        sources = "\n\n".join(f"Kaynak sayfa {n}:\n<<<\n{text}\n>>>" for n, text in pages)
    else:
        sources = f"Kaynak sayfa:\n<<<\n{pages[0][1]}\n>>>"
    return (
        f"Banka: {BANK_NAMES[bank]} | Rapor yılı: {year}\n\n{_INSTRUCTIONS}\n\n"
        f"Taslaklar:\n{listing}\n\n{sources}"
    )


def verdict_reason(item: CheckItem | None) -> str | None:
    """None when the item passes, else the drop reason."""
    if item is None:
        return "no check verdict"
    failed = [
        name
        for name, ok in (
            ("supported", item.supported),
            ("figure_ok", item.figure_ok is not False),
            ("basis_ok", item.basis_ok),
            ("self_contained", item.self_contained),
        )
        if not ok
    ]
    return f"check failed ({', '.join(failed)}): {item.reason}" if failed else None


def apply_check(
    drafts: Sequence[Draft], check: PageCheck | None
) -> tuple[list[Draft], list[tuple[Draft, str]]]:
    """(passing drafts with llm_checked=True, [(draft, reason)] dropped)."""
    by_index: dict[int, CheckItem] = {}
    if check is not None:
        for item in check.items:
            by_index.setdefault(item.index, item)
    passed: list[Draft] = []
    dropped: list[tuple[Draft, str]] = []
    for i, d in enumerate(drafts):
        reason = verdict_reason(by_index.get(i)) if check is not None else "check call failed"
        if reason is None:
            passed.append(d.model_copy(update={"llm_checked": True}))
        else:
            dropped.append((d, reason))
    return passed, dropped
