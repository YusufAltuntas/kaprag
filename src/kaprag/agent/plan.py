"""The router's model-facing parts: reply schemas, prompts and the report-year rule.

Three LLM steps use them (`kaprag.agent.graph` makes the calls):

- `classify` (Haiku) -> `RouteDecision`: `multi` for a comparison or calculation that
  needs figures for several banks and/or years, or arithmetic on figures; `simple`
  for everything else, including questions to abstain on or to clarify (the simple
  pipeline handles those). When in doubt, `simple`.
- `decompose` (Sonnet) -> `Plan`: atomic figure look-ups plus ONE operation from the
  closed set `OPERATIONS`, or `expressible: false` (the graph then falls back to the
  simple path). The schema enforces the operation's arity, references between the
  operation and the look-ups, the bank keys, the corpus report years and the cap of
  `MAX_LOOKUPS` look-ups.
- `synthesize` (Haiku) -> `SynthesizedAnswer`: the answer text only. The final
  `value` / `unit` come from `kaprag.agent.compute`, never from this model.

Operations (operands are look-up ids, in order; `kaprag.agent.compute` does the maths):

- `percent_change` [base, target]: (target - base) / |base| x 100, in %. Also "A,
  B'den yüzde kaç fazla" = percent_change [B, A].
- `difference` [a, b]: a - b, in the inputs' unit (TL, percentage points as %, adet).
- `ratio` [a, b]: a / b (times, unit-less). `share` [a, b]: a / b x 100, in %.
- `max` / `min` [...]: the winning look-up(s) and their figure.
- `rank` [...]: all look-ups ordered by value (`descending` true = largest first).

Report-year rule (eval/README.md section 4): a change between two consecutive years of
one bank reads BOTH figures from the later report (its comparative column).
`apply_report_year_rule` enforces this in code on the two-operand operations, so the
rule does not depend on the model following the prompt.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator

from kaprag.rag.query_filter import CORPUS_YEARS, BankKey

MAX_LOOKUPS = 8

OperationKind = Literal["percent_change", "difference", "ratio", "share", "max", "min", "rank"]
OPERATIONS: tuple[str, ...] = OperationKind.__args__
BINARY_OPERATIONS = frozenset({"percent_change", "difference", "ratio", "share"})

ReportYear = Literal[2023, 2024, 2025]  # == CORPUS_YEARS (tested)


# --- classify ----------------------------------------------------------------

CLASSIFY_SYSTEM_PROMPT = """\
Garanti BBVA, Akbank, İş Bankası ve Yapı Kredi'nin 2023-2025 faaliyet raporları \
üzerinde çalışan bir asistana gelen soruyu yönlendiriyorsun. Soruyu cevaplama; \
yalnızca hangi yoldan cevaplanacağına karar ver.

- `multi`: Cevap birden fazla bankanın ve/veya birden fazla yılın rakamlarını \
toplayıp karşılaştırmayı (en yüksek/en düşük, hangisi daha fazla, sıralama) ya da \
rakamlar üzerinde aritmetik işlem yapmayı (yüzde değişim, fark, oran, pay, "yüzde kaç \
fazla") gerektiriyorsa ve bankalar ile yıllar soruda açıkça belirtilmişse.
- `simple`: Diğer her şey: tek bir rakam ya da bilgi isteyen sorular, metin ya da \
özet isteyen sorular; banka ya da yıl belirtilmemiş sorular; dört banka dışında bir \
bankayı soran ya da raporlarda bulunması beklenmeyen bilgileri soran sorular.

Emin değilsen `simple` seç."""


class RouteDecision(BaseModel):
    """The classify call's reply."""

    route: Literal["simple", "multi"] = Field(
        description="`multi`: çok banka/yıl karşılaştırması ya da rakamlarla hesaplama; "
        "aksi halde `simple`."
    )
    reason: str = Field(default="", description="Kararın kısa gerekçesi.")


def classify_user_prompt(question: str) -> str:
    return f"Soru: {question}"


# --- decompose ----------------------------------------------------------------

_DECOMPOSE_SYSTEM_PROMPT = """\
Garanti BBVA, Akbank, İş Bankası ve Yapı Kredi'nin 2023-2025 faaliyet raporları \
üzerinde çalışan bir asistan için, karşılaştırma ya da hesaplama gerektiren bir \
soruyu bir plana çeviriyorsun. Soruyu cevaplama ve hiçbir rakamı tahmin etme. Plan:

1. `lookups`: Her biri TEK bir rakamı TEK bir raporda arayan atomik aramalar \
(en fazla {max_lookups} tane). Her arama için:
   - `id`: kısa benzersiz ad ("a", "b", "c", ...).
   - `bank`: "garanti" (Garanti BBVA), "akbank", "isbank" (Türkiye İş Bankası) ya da \
"yapikredi" (Yapı Kredi).
   - `fiscal_year`: rakamın ait olduğu yıl (ör. 2024 yıl sonu -> 2024).
   - `report_year`: rakamın okunacağı faaliyet raporunun yılı (2023, 2024 ya da 2025).
   - `metric`: aranan rakamın kısa adı (ör. "şube sayısı", "konsolide net kâr").
   - `basis`: soru konsolide ya da solo (bireysel) dediyse "konsolide" ya da "solo"; \
demediyse null.
   - `sub_question`: tek başına anlaşılır, Türkçe bir alt soru: bankanın tam adını, \
rakamın yılını, okunacağı raporu ve (soru belirttiyse) esası içerir. Örnek: \
"Akbank'ın 2024 faaliyet raporuna göre 2023 yıl sonu solo toplam kredileri ne kadardır?"
2. `operation`: aramalar üzerinde TEK bir işlem; `operands` arama `id`'lerinin sırası:
   - `percent_change` [taban, hedef]: tabandan hedefe yüzde değişim. "A, B'den yüzde \
kaç fazla/az" -> [B, A]; "2024'ten 2025'e yüzde kaç değişti" -> [2024, 2025].
   - `difference` [a, b]: a - b (para farkı TL, oran farkı yüzde puan).
   - `ratio` [a, b]: a / b ("kaç katı"). `share` [a, b]: a'nın b içindeki yüzde payı.
   - `max` / `min` [...]: en yüksek / en düşük olan (banka ya da yıl).
   - `rank` [...]: sıralama; `descending` true ise büyükten küçüğe.
   Her arama işlemde kullanılmalı; ikili işlemler tam iki arama alır, `max`, `min` ve \
`rank` en az iki.

Yıl kuralları:
- Tek bir yılın rakamı o yılın raporundan okunur (`report_year` = `fiscal_year`). \
Rakamın yılı için korpusta rapor yoksa (ör. 2022) bir sonraki yılın raporundan okunur.
- Aynı bankanın iki yılı arasındaki değişimde (yüzde değişim, fark) İKİ rakam da iki \
yıldan daha yeni olanın raporundan, karşılaştırmalı sütundan okunur: iki aramanın da \
`report_year` değeri daha sonraki yıldır (ör. 2024'ten 2025'e -> iki arama da \
`report_year` 2025).

Soru bu işlemlerden biriyle ifade edilemiyorsa, bankalar ya da yıllar açıkça \
belirtilmemişse, soru dört banka dışında bir bankayı soruyorsa ya da {max_lookups}'den \
fazla arama gerekiyorsa `expressible: false` yap, `reason` alanına nedenini yaz; \
`lookups` boş ve `operation` null kalsın."""


def decompose_system_prompt(max_lookups: int) -> str:
    return _DECOMPOSE_SYSTEM_PROMPT.format(max_lookups=max_lookups)


def decompose_user_prompt(question: str) -> str:
    return f"Soru: {question}"


class Lookup(BaseModel):
    """One atomic figure look-up: one bank, one figure, one report."""

    id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,16}$", description='Kısa benzersiz ad ("a").')
    bank: BankKey = Field(description="Banka anahtarı.")
    fiscal_year: Annotated[int, Field(strict=True, ge=2000, le=max(CORPUS_YEARS))] = Field(
        description="Rakamın ait olduğu yıl."
    )
    report_year: ReportYear = Field(description="Rakamın okunacağı raporun yılı.")
    metric: str = Field(min_length=1, description="Aranan rakamın kısa adı.")
    basis: Literal["konsolide", "solo"] | None = Field(
        default=None, description="Soru belirttiyse esas; yoksa null."
    )
    sub_question: str = Field(min_length=1, description="Tek başına anlaşılır Türkçe alt soru.")

    @model_validator(mode="after")
    def _check(self) -> Lookup:
        if self.fiscal_year > self.report_year:
            raise ValueError(
                f"look-up {self.id!r}: fiscal_year {self.fiscal_year} is after "
                f"report_year {self.report_year}"
            )
        return self


class Operation(BaseModel):
    kind: OperationKind = Field(description="İşlem.")
    operands: list[str] = Field(min_length=2, description="Arama id'leri, sırayla.")
    descending: bool = Field(default=True, description="`rank` için: büyükten küçüğe.")

    @model_validator(mode="after")
    def _check(self) -> Operation:
        if len(set(self.operands)) != len(self.operands):
            raise ValueError(f"operation operands repeat: {self.operands}")
        if self.kind in BINARY_OPERATIONS and len(self.operands) != 2:
            raise ValueError(f"{self.kind} takes exactly 2 operands, got {len(self.operands)}")
        return self


class Plan(BaseModel):
    """The decompose call's reply."""

    expressible: bool = Field(description="Soru bu işlemlerle ifade edilebiliyor mu.")
    reason: str | None = Field(default=None, description="İfade edilemiyorsa nedeni.")
    lookups: list[Lookup] = Field(default_factory=list, max_length=MAX_LOOKUPS)
    operation: Operation | None = None

    @model_validator(mode="after")
    def _check(self) -> Plan:
        if not self.expressible:
            return self
        if self.operation is None:
            raise ValueError("an expressible plan needs an operation")
        ids = [lookup.id for lookup in self.lookups]
        if len(set(ids)) != len(ids):
            raise ValueError(f"look-up ids repeat: {ids}")
        if unknown := [o for o in self.operation.operands if o not in ids]:
            raise ValueError(f"operands {unknown} are not look-up ids {ids}")
        if unused := [i for i in ids if i not in self.operation.operands]:
            raise ValueError(f"look-ups {unused} are not used by the operation")
        return self

    def lookup(self, lookup_id: str) -> Lookup:
        return next(lookup for lookup in self.lookups if lookup.id == lookup_id)


def apply_report_year_rule(plan: Plan) -> tuple[Plan, list[dict[str, object]]]:
    """Enforce section 4 on a two-operand operation: when both operands are the same
    bank in two consecutive fiscal years, both are read from the later year's report.
    Returns (plan, fixes); each fix records a changed look-up's old and new
    report_year. Other plans are returned unchanged."""
    op = plan.operation
    if not plan.expressible or op is None or op.kind not in BINARY_OPERATIONS:
        return plan, []
    first, second = (plan.lookup(i) for i in op.operands)
    years = sorted((first.fiscal_year, second.fiscal_year))
    later = years[1]
    if first.bank != second.bank or years[1] - years[0] != 1 or later not in CORPUS_YEARS:
        return plan, []
    fixes: list[dict[str, object]] = []
    lookups = []
    for lookup in plan.lookups:
        if lookup.id in op.operands and lookup.report_year != later:
            fixes.append({"id": lookup.id, "from": lookup.report_year, "to": later})
            lookup = lookup.model_copy(update={"report_year": later})
        lookups.append(lookup)
    return plan.model_copy(update={"lookups": lookups}), fixes


# --- synthesize ---------------------------------------------------------------

SYNTHESIZE_SYSTEM_PROMPT = """\
Garanti BBVA, Akbank, İş Bankası ve Yapı Kredi'nin faaliyet raporları üzerinde \
çalışan bir finansal analiz asistanısın. Sana bir soru, kaynaklardan bulunmuş rakamlar \
ve bu rakamlarla kodla yapılmış hesabın sonucu verilecek. Bunlardan soruya kısa, net \
bir cevap yaz.

Kurallar:
1. Cevabı sorunun dilinde yaz.
2. Hesap yapma ve rakamları değiştirme: sonucu ve rakamları sana verildiği gibi kullan; \
sana verilmeyen yeni bir sayı (fark, oran, yuvarlanmış değer vb.) yazma.
3. Karşılaştırma sorusunda kazanan bankayı (ya da yılı) açıkça söyle ve rakamları ver.
4. Hesaplama sorusunda sonucu ve hangi rakamlardan hesaplandığını söyle; değişim \
negatifse azalış, pozitifse artış de.
5. Rakamlar konsolide ya da solo esaslıysa bunu söyle; soru esası belirtmiyor ve \
konsolide rakam kullanıldıysa konsolideyi varsaydığını açıkça belirt."""


class SynthesizedAnswer(BaseModel):
    """The synthesize call's reply: the answer text only."""

    answer: str = Field(min_length=1, description="Sorunun dilinde kısa cevap.")


def synthesize_user_prompt(question: str, facts: str) -> str:
    return f"Soru: {question}\n\n{facts}"
