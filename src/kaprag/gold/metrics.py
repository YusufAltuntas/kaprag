"""The standard metrics the probe stage looks up for every (bank, year) report.

Each metric is a headline figure all four banks' integrated annual reports state.
Phrasing helpers build analyst-style Turkish questions in code (probe singles and the
comparisons / calculations composed from them), so the wording never copies a table's
row label.
"""

from __future__ import annotations

from dataclasses import dataclass

from kaprag.gold.draft import BANK_GENITIVE, BANK_NAMES


@dataclass(frozen=True)
class Metric:
    id: str
    unit: str  # canonical unit: TL / % / adet
    type: str  # gold type of the single question: tablo / olgusal
    needs_basis: bool  # figure differs between consolidated and solo statements
    stock: bool  # year-end balance (True) or a flow over the year (False)
    description: str  # what to extract, for the extraction prompt
    query: str  # retrieval query (the bank and year are added)
    noun: str  # possessive noun phrase: "net kârı", "şube sayısı"
    single: str  # single question template: {name} {gen} {year} {adv} {adj}


METRICS: tuple[Metric, ...] = (
    Metric(
        id="net_profit",
        unit="TL",
        type="tablo",
        needs_basis=True,
        stock=False,
        description="yılın net dönem kârı (vergi sonrası, ana ortaklık ve azınlık payları dahil "
        "toplam net dönem kârı/zararı)",
        query="net dönem kârı gelir tablosu",
        noun="net kârı",
        single="{name}, {year} yılında {adv}ne kadar net kâr elde etti?",
    ),
    Metric(
        id="total_assets",
        unit="TL",
        type="tablo",
        needs_basis=True,
        stock=True,
        description="yıl sonu toplam aktifler (bilanço aktif toplamı)",
        query="aktif büyüklüğü toplam aktifler bilanço",
        noun="aktif büyüklüğü",
        single="{gen} {year} yıl sonundaki {adj}aktif büyüklüğü ne kadardı?",
    ),
    Metric(
        id="equity",
        unit="TL",
        type="tablo",
        needs_basis=True,
        stock=True,
        description="yıl sonu toplam özkaynaklar (bilançodaki özkaynaklar toplamı)",
        query="özkaynaklar toplamı bilanço",
        noun="özkaynak büyüklüğü",
        single="{gen} {year} yıl sonundaki {adj}özkaynak büyüklüğü ne kadardı?",
    ),
    Metric(
        id="car",
        unit="%",
        type="tablo",
        needs_basis=True,
        stock=True,
        description="yıl sonu sermaye yeterliliği oranı (toplam SYR, yüzde)",
        query="sermaye yeterliliği oranı",
        noun="sermaye yeterliliği oranı",
        single="{gen} {year} yıl sonundaki {adj}sermaye yeterliliği oranı yüzde kaçtı?",
    ),
    Metric(
        id="branches",
        unit="adet",
        type="olgusal",
        needs_basis=False,
        stock=True,
        description="yıl sonu yurt içi + yurt dışı toplam şube sayısı (Banka'nın kendi şubeleri; "
        "iştirak/grup şirketleri hariç)",
        query="şube sayısı yurt içi yurt dışı şube",
        noun="şube sayısı",
        single="{name}, {year} yıl sonunda kaç şubeyle hizmet veriyordu?",
    ),
    Metric(
        id="employees",
        unit="adet",
        type="olgusal",
        needs_basis=False,
        stock=True,
        description="yıl sonu toplam çalışan (personel) sayısı (Banka'nın kendisi; Grup/konsolide "
        "personel sayısı değil)",
        query="çalışan sayısı personel sayısı",
        noun="çalışan sayısı",
        single="{gen} {year} yıl sonunda kaç çalışanı vardı?",
    ),
)
METRICS_BY_ID: dict[str, Metric] = {m.id: m for m in METRICS}

BASIS_ADJ = {"konsolide": "konsolide ", "solo": "solo ", None: ""}
BASIS_ADV = {"konsolide": "konsolide bazda ", "solo": "solo bazda ", None: ""}


def single_question(metric: Metric, bank: str, year: int, basis: str | None) -> str:
    """Analyst-style question for one metric of one bank and year."""
    return metric.single.format(
        name=BANK_NAMES[bank],
        gen=BANK_GENITIVE[bank],
        year=year,
        adj=BASIS_ADJ[basis],
        adv=BASIS_ADV[basis],
    )


def time_phrase(metric: Metric, year: int) -> str:
    """'2024 yıl sonu itibarıyla' for balances, '2024 yılında' for flows."""
    return f"{year} yıl sonu itibarıyla" if metric.stock else f"{year} yılında"
