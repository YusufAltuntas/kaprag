"""Step 2: one Sonnet call per sampled page drafts that page's questions; code validates.

The page (evidence) is known by construction - the model is never asked for page
numbers. The model returns figures as written plus their unit text; `figures.read_figure`
recomputes the canonical value. `hesaplama` drafts give the current- and prior-year
figures of one row of a comparative table, and the change is computed in Python
(`derive.yoy_expected`), never by the model.
"""

from __future__ import annotations

from collections.abc import Set
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel

from kaprag.eval.schema import Evidence, Expected
from kaprag.gold.derive import yoy_expected
from kaprag.gold.draft import BANK_NAMES, TOLERANCE_BY_UNIT, Draft
from kaprag.gold.figures import FigureError, UnitError, read_figure
from kaprag.gold.text import bases_mentioned, copied_run, mentions_bank, years_mentioned

PAGE_TYPES = ("olgusal", "tablo", "hesaplama", "sentez")


class DraftOut(BaseModel):
    type: Literal["olgusal", "tablo", "hesaplama", "sentez"]
    question: str
    answer: str
    year: int
    basis: Literal["konsolide", "solo"] | None
    figure: str | None
    unit_text: str | None
    value: int | float | None
    unit: Literal["TL", "%", "adet"] | None
    prior_figure: str | None
    prior_value: int | float | None
    key_points: list[str]


class PageDrafts(BaseModel):
    questions: list[DraftOut]


@dataclass(frozen=True)
class PageTask:
    doc_id: str
    bank: str
    year: int
    page: int
    requests: dict[str, int]  # type -> count

    @property
    def key(self) -> str:
        return f"page:{self.doc_id}:{self.page}"


SYSTEM = (
    "Sen Türk bankacılık sektörünü iyi bilen titiz bir finansal analistsin. Bir soru-cevap "
    "sisteminin değerlendirme seti için banka faaliyet raporlarının tek bir sayfasından "
    "soru-cevap taslakları hazırlıyorsun. Yalnızca sayfada açıkça yazan bilgiyi kullanırsın; "
    "emin olmadığın hiçbir şeyi uydurmazsın."
)

_TYPE_HELP = {
    "tablo": "sayfadaki bir tablodan okunan TEK bir sayısal değer (tutar, oran ya da adet)",
    "olgusal": "sayfadaki tek bir olgu (bir sayı, isim, tarih, hedef, ürün, oran...)",
    "hesaplama": (
        "bir tablonun aynı satırındaki cari yıl ({year}) ve önceki yıl ({prior}) değerleri; soru "
        "iki yıl arasındaki değişimi sorar: tutar/adet için 'yüzde kaç değişti', oran (%) için "
        "'kaç puan değişti'. Hesabı SEN yapma; answer alanına iki değeri yaz"
    ),
    "sentez": (
        "sayfadaki birden çok bilgiyi birleştirmeyi gerektiren açık uçlu bir soru (strateji, "
        "hedefler, öncelikler, gelişmeler...); key_points: cevabın kapsaması gereken 3-5 kısa madde"
    ),
}

_RULES = """Kurallar:
1. Analist dili: Soruyu bir analistin ya da yatırımcının doğal olarak soracağı gibi yaz; \
sayfadaki ifadeleri, satır başlıklarını ve cümleleri KOPYALAMA ("dönem net kârı" yerine "ne \
kadar kâr etti" gibi). Sayfadaki 6 kelimelik bir diziyi aynen tekrar eden soru reddedilir.
2. Kendi kendine yeterli soru: Soruda banka adı ({bank}) ve yıl ({year}) açıkça geçmeli; soru \
hangi sayfadan geldiği bilinmeden cevaplanabilmeli ("bu tabloda", "yukarıdaki" gibi ifadeler yok).
3. Esas (basis): Finansal tutar ve oranlarda sayfa (başlık, tablo adı, dipnot başlığı) rakamın \
konsolide mi ("KONSOLİDE") yoksa solo mu ("KONSOLİDE OLMAYAN", banka solo) olduğunu açıkça \
belirtiyorsa basis'i doldur ve soruda "konsolide" ya da "solo" diye yaz. Sayfa bunu belirtmiyorsa \
basis null olsun ve soruda esas belirtme. TAHMİN ETME.
4. Yıl: Tekil sorular rapor yılına ({year}) ait değerleri sorar (year={year}); önceki yıl \
sütunundan tekil soru sorma.
5. Önem: Sorular bankanın kendisi hakkında olsun (finansal sonuçları, faaliyetleri, \
stratejisi, hedefleri, müşterileri, çalışanları...); raporda geçen genel makroekonomik ya da \
sektörel değerlendirmeler (piyasa faizi, enflasyon, büyüme tahmini...) hakkında soru üretme. Bir \
analistin gerçekten merak edeceği kalemleri seç (ana tutarlar, toplamlar, temel oranlar, önemli \
olgular); dipnot ve risk tablolarındaki ayrıntılı alt kırılımları ve satır başlığı listelerini \
soru yapma. Sayfada uygun içerik yoksa istenenden az soru üret; boş liste de olabilir. Uydurma yok.
6. Sayılar (tablo, sayısal olgusal, hesaplama):
   - figure: sayı, sayfada yazdığı gibi (ör. "1.073.924", "(3.572.806)", "12,5", "%17,3"). \
Yalnızca sayının kendisi; birim kelimesi ekleme.
   - unit_text: o sayıya uygulanan birim, sayfadaki ifadesiyle ("bin TL", "milyon TL", \
"milyar TL", "TL", "%", "adet", "kişi", "şube"...). Para tutarlarında çarpanı tablonun \
başlığından ya da "(Tutarlar aksi belirtilmedikçe bin Türk Lirası ...)" notundan oku. Çarpan \
(bin/milyon/milyar) sayfada yazmıyorsa o tablodan para sorusu ÜRETME; birimi tahmin etme.
   - value: normalize değer. Para: TL cinsinden tam sayı, çarpan uygulanmış \
(1.073.924 bin TL -> 1073924000; 45,2 milyar TL -> 45200000000). Oran: yüzde sayısı \
(%17,3 -> 17.3). Adet: tam sayı. unit: "TL", "%" ya da "adet".
   - hesaplama'da figure/value cari yıl ({year}) değeri, prior_figure/prior_value önceki yıl \
({prior}) değeri; ikisi aynı birimde. Soruda iki yıl da geçmeli.
   - Sayısal olmayan sorularda figure, unit_text, value, unit, prior_* null.
7. answer: Kısa, sayfaya dayalı cevap cümlesi (sayısal ise sayfadaki birimle birlikte).
8. key_points yalnızca sentez için; diğer tiplerde boş liste."""


def build_prompt(task: PageTask, page_text: str) -> str:
    prior = task.year - 1
    lines = [
        f"Banka: {BANK_NAMES[task.bank]} | Rapor: {task.year} yılı faaliyet raporu "
        f"(rapor yılı {task.year}) | Bu, raporun tek bir sayfası.",
        "",
        "İstenen sorular (her tipten en fazla bu kadar):",
    ]
    for type_, n in task.requests.items():
        if n <= 0:
            continue
        help_text = _TYPE_HELP[type_].format(year=task.year, prior=prior)
        lines.append(f"- {type_}: {n} adet - {help_text}")
    lines += ["", _RULES.format(bank=BANK_NAMES[task.bank], year=task.year, prior=prior), ""]
    lines += ["Sayfa içeriği (Markdown; tablolar Markdown tablo):", "<<<", page_text, ">>>"]
    return "\n".join(lines)


def _numeric_expected(out: DraftOut, numbers: Set[Decimal], page_text: str) -> Expected:
    value, unit = read_figure(out.figure, out.unit_text, out.value, out.unit, numbers, page_text)
    return Expected(answer=out.answer, value=value, unit=unit, tolerance=TOLERANCE_BY_UNIT[unit])


def validate_draft(
    out: DraftOut,
    task: PageTask,
    index: int,
    page_text: str,
    numbers: Set[Decimal],
) -> Draft:
    """A Draft from the model's output, or FigureError / ValueError with the drop reason."""
    if task.requests.get(out.type, 0) <= 0:
        raise ValueError(f"type {out.type!r} was not requested")
    if out.year != task.year:
        raise ValueError(f"year {out.year} is not the report year {task.year}")
    q = out.question
    if not mentions_bank(q, task.bank):
        raise ValueError("question does not name the bank")
    years = [task.year - 1, task.year] if out.type == "hesaplama" else [task.year]
    if not set(years) <= years_mentioned(q):
        raise ValueError(f"question does not name the year(s) {years}")
    named = bases_mentioned(q)
    if out.basis is None and named:
        raise ValueError(f"question names basis {sorted(named)} but basis is null")
    if out.basis is not None and named != {out.basis}:
        raise ValueError(f"basis {out.basis!r} but question names {sorted(named) or 'none'}")
    if run := copied_run(q, page_text):
        raise ValueError(f"question copies page wording: {run!r}")
    if not out.answer.strip():
        raise ValueError("empty answer")

    notes = ""
    key_points: list[str] = []
    quote = out.figure
    if out.type == "sentez":
        key_points = [k.strip() for k in out.key_points if k.strip()]
        if len(key_points) < 2:
            raise ValueError("sentez needs at least 2 key points")
        expected = Expected(answer=out.answer, value=None, unit=None, tolerance=None)
    elif out.type == "tablo":
        expected = _numeric_expected(out, numbers, page_text)
    elif out.type == "olgusal":
        if out.figure is None:
            expected = Expected(answer=out.answer, value=None, unit=None, tolerance=None)
        else:
            try:
                expected = _numeric_expected(out, numbers, page_text)
            except UnitError as exc:  # on the page, but not TL / % / adet: judged as text
                expected = Expected(answer=out.answer, value=None, unit=None, tolerance=None)
                notes = f"figure {out.figure!r} on the page; scored as text ({exc})"
    else:  # hesaplama
        current = _numeric_expected(out, numbers, page_text)
        prior_value, prior_unit = read_figure(
            out.prior_figure, out.unit_text, out.prior_value, out.unit, numbers, page_text
        )
        unit = current.unit
        if unit in ("TL", "%") and out.basis is None:
            raise ValueError("calculation on a money/ratio figure of unknown basis")
        wanted = "puan" if unit == "%" else "yüzde"
        if wanted not in q.lower():
            raise ValueError(f"calculation question should ask '{wanted}'")
        expected = yoy_expected(
            Decimal(str(prior_value)), Decimal(str(current.value)), unit, task.year - 1, task.year
        )
        quote = f"{task.year}: {out.figure}; {task.year - 1}: {out.prior_figure} ({out.unit_text})"
        notes = "change computed in Python from the page's current and prior-year columns"
    if expected.unit in ("TL", "%") and out.type != "hesaplama" and out.basis is None:
        notes = "basis (konsolide/solo) not stated on the page"
    return Draft(
        key=f"{task.key}:{index}",
        source="page",
        type=out.type,
        question=q.strip(),
        banks=[task.bank],
        years=years,
        basis=out.basis,
        expected_behavior="answer",
        expected=expected,
        evidence=[Evidence(doc=task.doc_id, page=task.page)],
        key_points=key_points,
        notes=notes,
        quote=quote if out.type != "sentez" else None,
    )


def validate_page(
    result: PageDrafts, task: PageTask, page_text: str, numbers: Set[Decimal]
) -> tuple[list[Draft], list[tuple[str, str, DraftOut]]]:
    """(valid drafts, [(key, reason, output)] drops) for one page's model output."""
    drafts: list[Draft] = []
    drops: list[tuple[str, str, DraftOut]] = []
    seen_questions: set[str] = set()
    for i, out in enumerate(result.questions):
        key = f"{task.key}:{i}"
        try:
            draft = validate_draft(out, task, i, page_text, numbers)
        except (FigureError, ValueError) as exc:
            drops.append((key, str(exc), out))
            continue
        if draft.question in seen_questions:
            drops.append((key, "duplicate question on the page", out))
            continue
        seen_questions.add(draft.question)
        drafts.append(draft)
    return drafts, drops
