"""Metadata filter from the question: one LLM call parses the corpus banks and fiscal
years the question is about; retrieval is then restricted to them in SQL.

Per question (config block `query_filter`, see `kaprag.rag.config`):

1. One schema-validated call (`ParsedQuery`) tagged `{**tags, "step": "parse_query"}`,
   so the eval runner attributes its cost and latency to the question. `banks` holds
   only the four corpus bank keys (a bank outside the corpus, e.g. Ziraat, yields no
   key, never a guess); `years` holds the years written in the question.
2. The filter (`metadata_filter`): `bank IN (banks)` when `banks` is non-empty, and
   `year IN (years ∩ CORPUS_YEARS)` when that intersection is non-empty; no condition
   otherwise. Values are validated against the bank keys / integer years and bound as
   parameters (`kaprag.retrieval.SqlFilter`), never written into the SQL text. Both
   retrieval arms apply it before ranking.
3. A failed parse (`LLMCallError`: backend error, invalid JSON after the client's one
   retry) or a parse with nothing usable means no filter: the question is retrieved
   unfiltered and answered as usual; the failure is recorded in the debug output.

Year semantics (eval/README.md section 4): a single-year figure is answered from that
year's report, a year-over-year change ("2024'ten 2025'e") from the most recent
report's comparative columns. `year IN (stated years)` keeps the most recent stated
year's report in scope in both cases, so there is no further year logic. A year with
no report in the corpus (e.g. 2022, present only as comparative columns of the 2023
reports) drops out of the intersection; if no stated year is left, no year condition
is applied, on purpose.

A filter that leaves fewer candidates than the context needs is not widened: the
retrieved list is what the filter gives (an empty one reaches generation as an empty
context, where the generation prompt abstains).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from kaprag.corpus import BANK_NAMES
from kaprag.llm import LLMCallError, LLMClient
from kaprag.retrieval import SqlFilter

# Fiscal years with a report in the corpus (data/manifest.csv; pinned by a test).
CORPUS_YEARS: tuple[int, ...] = (2023, 2024, 2025)

BankKey = Literal["garanti", "akbank", "isbank", "yapikredi"]  # == BANK_NAMES keys (tested)

SYSTEM_PROMPT = """\
Bir soruyu Garanti BBVA, Akbank, İş Bankası ve Yapı Kredi'nin faaliyet raporlarında \
arama yapmak için ayrıştırıyorsun. Sorunun hangi bankalar ve hangi yıllar hakkında \
olduğunu çıkar; soruyu cevaplama.

Kurallar:
1. `banks`: soruda adı geçen bankalardan yalnızca şu dördünü, şu anahtarlarla yaz: \
Garanti BBVA (Garanti) -> "garanti"; Akbank -> "akbank"; Türkiye İş Bankası \
(İş Bankası, İşbank) -> "isbank"; Yapı Kredi (Yapı ve Kredi Bankası) -> "yapikredi". \
Bu dördünden biri olmayan bir banka (ör. Ziraat, Halkbank) için hiçbir şey yazma. \
Soruda bu dört bankadan hiçbiri adıyla geçmiyorsa liste boş kalır; tahmin etme.
2. `years`: soruda açıkça yazılan yılları dört basamaklı tam sayı olarak yaz \
(ör. "2024'ten 2025'e" -> [2024, 2025]). Soruda yıl yazmıyorsa liste boş kalır; \
"geçen yıl", "son yıl" gibi ifadelerden yıl çıkarma."""


class ParsedQuery(BaseModel):
    """The parse call's reply."""

    banks: list[BankKey] = Field(
        default_factory=list,
        description="Soruda adı geçen korpus bankalarının anahtarları; yoksa boş.",
    )
    years: list[Annotated[int, Field(strict=True)]] = Field(
        default_factory=list, description="Soruda açıkça yazılan yıllar; yoksa boş."
    )


@dataclass(frozen=True)
class QueryFilterOutcome:
    """The filter for retrieval (None = unfiltered) and what to record in the debug
    output: `parsed` (the model's reply, None on failure), `applied` (the filter's
    banks/years, None when no filter) and `error` (the failure, or None)."""

    where: SqlFilter | None
    debug: dict[str, Any]


def build_user_prompt(question: str) -> str:
    return f"Soru: {question}"


def filter_values(parsed: ParsedQuery) -> tuple[list[str], list[int]]:
    """(banks, years) to filter on: distinct, in corpus order; years outside
    `CORPUS_YEARS` dropped."""
    banks = [b for b in BANK_NAMES if b in parsed.banks]
    years = [y for y in CORPUS_YEARS if y in parsed.years]
    return banks, years


def metadata_filter(banks: Sequence[str], years: Sequence[int]) -> SqlFilter | None:
    """`bank IN banks AND year IN years`, each only when non-empty; None for no
    condition. Values are checked here, so nothing unvalidated reaches the SQL."""
    if unknown := [b for b in banks if b not in BANK_NAMES]:
        raise ValueError(f"unknown bank keys {unknown!r}")
    if bad := [y for y in years if type(y) is not int]:
        raise ValueError(f"years must be integers, got {bad!r}")
    where = SqlFilter()
    if banks:
        where = where.any_of("bank", list(banks))
    if years:
        where = where.any_of("year", list(years))
    return where or None


async def parse_query_filter(
    client: LLMClient, question: str, *, model: str, tags: dict[str, Any]
) -> QueryFilterOutcome:
    """Parse the question and build its filter; a failed call means no filter."""
    try:
        result = await client.complete(
            SYSTEM_PROMPT,
            build_user_prompt(question),
            model,
            schema=ParsedQuery,
            tags={**tags, "step": "parse_query"},
        )
    except LLMCallError as exc:
        return QueryFilterOutcome(None, {"parsed": None, "applied": None, "error": str(exc)})
    parsed: ParsedQuery = result.parsed
    banks, years = filter_values(parsed)
    where = metadata_filter(banks, years)
    return QueryFilterOutcome(
        where,
        {
            "parsed": parsed.model_dump(mode="json"),
            "applied": {"banks": banks, "years": years} if where is not None else None,
            "error": None,
        },
    )
