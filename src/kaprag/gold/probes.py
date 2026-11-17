"""Step 4 input: metric probes - the same standard metrics located in every report.

Per document: retrieve the top fused passages for each metric (restricted to that
document), then ONE extraction call reads every metric's report-year figure from the
passages. Code resolves the evidence page (the page of the cited passage whose text
holds the figure), recomputes the canonical value, and words the question from a
template (`metrics.single_question`). A second call per document checks every probe
against its rendered evidence page(s). Each checked probe is a `tablo`/`olgusal`
single and the raw material of `derive.compose_from_probes`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel

from kaprag.eval.schema import Evidence, Expected
from kaprag.gold.draft import BANK_NAMES, TOLERANCE_BY_UNIT, Draft
from kaprag.gold.figures import FigureError, figure_number, read_figure
from kaprag.gold.metrics import Metric, single_question
from kaprag.gold.pages import DocPages
from kaprag.retrieval import SearchHit

PASSAGES_PER_METRIC = 5


class ProbeOut(BaseModel):
    metric: str
    found: bool
    passage: int | None
    figure: str | None
    components: list[str]
    unit_text: str | None
    value: int | float | None
    unit: Literal["TL", "%", "adet"] | None
    basis: Literal["konsolide", "solo"] | None
    answer: str | None


class ProbeExtraction(BaseModel):
    items: list[ProbeOut]


SYSTEM = (
    "Sen banka faaliyet raporlarından rakam çıkaran titiz bir finansal analistsin. Yalnızca "
    "verilen pasajlarda açıkça yazan rakamları kullanırsın; bulamazsan found=false dersin."
)


def metric_query(metric: Metric, bank: str, year: int) -> str:
    basis = " konsolide" if metric.needs_basis else ""
    return f"{BANK_NAMES[bank]} {year} {metric.query}{basis}"


def collect_passages(
    hits_by_metric: dict[str, list[SearchHit]], per_metric: int = PASSAGES_PER_METRIC
) -> list[SearchHit]:
    """The union of each metric's top hits, deduplicated, in metric then rank order."""
    seen: dict[str, SearchHit] = {}
    for hits in hits_by_metric.values():
        for hit in hits[:per_metric]:
            seen.setdefault(hit.chunk_id, hit)
    return list(seen.values())


def build_prompt(
    bank: str, year: int, metrics: Sequence[Metric], passages: Sequence[SearchHit]
) -> str:
    wanted = "\n".join(f"- {m.id}: {m.description} (birim: {m.unit})" for m in metrics)
    listing = "\n\n".join(
        f"[P{i}] (sayfa {h.page_start}-{h.page_end})\n{h.text}" for i, h in enumerate(passages, 1)
    )
    return f"""Banka: {BANK_NAMES[bank]} | Rapor: {year} yılı faaliyet raporu

Aşağıdaki metriklerin {year} yılına ait değerini (bakiye kalemlerinde {year} yıl sonu, \
akım kalemlerinde {year} yılının tamamı) pasajlardan bul. Her metrik için items listesine \
bir kayıt yaz:
- found: değer pasajlarda açıkça yazıyorsa true; yoksa false (diğer alanlar null).
- passage: rakamın geçtiği pasaj numarası (P3 için 3).
- figure: rakam, pasajda yazdığı gibi (ör. "1.073.924", "%17,3", "4 trilyon 548 milyar"); \
birim kelimesi (TL, kişi, şube) ekleme.
- components: Pasaj toplamı tek sayı olarak vermiyor ama parçalarını veriyorsa (ör. "693 yurt \
içi ve 1 yurt dışı şube"), figure null olsun ve parçaları pasajda yazdığı gibi bu listeye yaz \
(["693", "1"]); value'ya toplamı yaz (toplamı kod da hesaplar). Aksi halde boş liste.
- unit_text: o rakama uygulanan birim, pasajdaki ifadesiyle ("bin TL", "milyon TL", \
"milyar TL", "TL", "%", "adet", "şube", "kişi"...). Tablolarda çarpanı tablo başlığından ya da \
"(Tutarlar aksi belirtilmedikçe bin Türk Lirası ...)" notundan oku.
- value: normalize değer (para: TL cinsinden tam sayı, çarpan uygulanmış; oran: yüzde \
sayısı; adet: tam sayı); unit: "TL", "%" ya da "adet".
- basis: rakamın konsolide mi solo mu ("konsolide olmayan") olduğu pasajda açıkça \
belirtiliyorsa doldur; belirtilmiyorsa null. Tahmin etme. Hem konsolide hem solo rakam \
varsa konsolide olanı seç; aynı rakam birden çok pasajda geçiyorsa esasın açıkça yazdığı \
pasajı (ör. konsolide finansal tablo ya da konsolide özet) tercih et.
- answer: kısa cevap cümlesi.
Önceki yıl sütununu ({year - 1}) ya da çok yıllık özetteki başka yılları KULLANMA. Benzer ama \
farklı kalemleri (ör. net kâr yerine vergi öncesi kâr, toplam yerine alt kalem) KULLANMA.

Metrikler:
{wanted}

Pasajlar:
{listing}"""


def resolve_page(figure: str, hit: SearchHit, pages: DocPages) -> int:
    """The first page of the passage's span that has the figure's (first) number;
    FigureError if none."""
    number = figure_number(figure)
    for page in range(hit.page_start, hit.page_end + 1):
        numbers = pages.numbers(page)
        if number in numbers or abs(number) in numbers:
            return page
    raise FigureError(f"figure {figure!r} not on pages {hit.page_start}-{hit.page_end}")


def validate_probe(
    out: ProbeOut,
    metric: Metric,
    doc_id: str,
    bank: str,
    year: int,
    passages: Sequence[SearchHit],
    pages: DocPages,
) -> Draft:
    """A probe Draft, or FigureError / ValueError with the drop reason."""
    if not out.found:
        raise ValueError("not found in passages")
    if out.passage is None or not 1 <= out.passage <= len(passages):
        raise ValueError(f"invalid passage number {out.passage}")
    parts = [out.figure] if out.figure else out.components
    if not parts:
        raise ValueError("no figure given")
    page = resolve_page(parts[0], passages[out.passage - 1], pages)
    page_text = pages.render(page) + "\n" + pages.plain.get(page, "")
    value, unit = read_figure(
        out.figure,
        out.unit_text,
        out.value,
        out.unit,
        pages.numbers(page),
        page_text,
        components=out.components,
    )
    if unit != metric.unit:
        raise ValueError(f"unit {unit} is not the metric's unit {metric.unit}")
    basis = out.basis if metric.needs_basis else None
    notes = "probe fact"
    if metric.needs_basis and basis is None:
        notes += "; basis (konsolide/solo) not stated in the passage - not used in comparisons"
    return Draft(
        key=f"probe:{doc_id}:{metric.id}",
        source="probe",
        type=metric.type,
        question=single_question(metric, bank, year, basis),
        banks=[bank],
        years=[year],
        basis=basis,
        expected_behavior="answer",
        expected=Expected(
            answer=out.answer or f"{' + '.join(parts)} {out.unit_text or ''}".strip(),
            value=value,
            unit=unit,
            tolerance=TOLERANCE_BY_UNIT[unit],
        ),
        evidence=[Evidence(doc=doc_id, page=page)],
        notes=notes,
        metric=metric.id,
        quote=f"{' + '.join(parts)} ({out.unit_text})",
    )
