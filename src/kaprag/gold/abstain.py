"""Step 5: `cevapsiz` drafts (expected behaviour `abstain`), each checked against the corpus.

Three kinds (eval/README.md §7 step 5): a bank outside the corpus, a false premise (built
from checked probe facts, e.g. a loss for a bank that made a profit), and a plausible
metric the report does not state. Every candidate is searched in the corpus (dense +
full-text, RRF) and an LLM judge decides from the top passages whether it is in fact
answerable; only "not in corpus" (and, for false premises, "premise contradicted")
survive. The corpus holds 2022 comparatives and multi-year summaries, so a year outside
2023-2025 is not by itself a reason to keep a candidate.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel

from kaprag.eval.schema import YEAR_MAX, YEAR_MIN, Evidence, Expected
from kaprag.gold.derive import format_value
from kaprag.gold.draft import BANK_NAMES, Draft
from kaprag.gold.text import banks_mentioned, years_mentioned
from kaprag.retrieval import SearchHit

KINDS = ("korpus_disi_banka", "yanlis_oncul", "olmayan_bilgi")
PASSAGES = 6

Kind = Literal["korpus_disi_banka", "yanlis_oncul", "olmayan_bilgi"]
CorpusBank = Literal["garanti", "akbank", "isbank", "yapikredi"]


class AbstainOut(BaseModel):
    kind: Kind
    question: str
    banks: list[CorpusBank]
    years: list[int]
    why: str


class AbstainDrafts(BaseModel):
    questions: list[AbstainOut]


class AbstainVerdict(BaseModel):
    index: int
    verdict: Literal["answerable", "partly_answerable", "false_premise", "not_in_corpus"]
    evidence_passages: list[int]
    reason: str


class AbstainJudgement(BaseModel):
    items: list[AbstainVerdict]


DRAFT_SYSTEM = (
    "Sen bir soru-cevap sisteminin halüsinasyon ve reddetme davranışını ölçen değerlendirme "
    "seti hazırlayan bir finansal analistsin."
)
JUDGE_SYSTEM = (
    "Sen titiz bir denetçisin. Bir sorunun verilen rapor pasajlarından cevaplanıp "
    "cevaplanamayacağına karar verirsin; şüphede kalırsan 'partly_answerable' dersin."
)


def build_draft_prompt(per_kind: int, facts: Sequence[Draft]) -> str:
    fact_lines = "\n".join(
        f"- {BANK_NAMES[f.banks[0]]} {f.years[0]} {f.metric} ({f.basis or 'esas belirsiz'}): "
        f"{format_value(f.expected.value, f.expected.unit)}"
        for f in facts
        if f.expected.value is not None
    )
    return f"""Korpus: Garanti BBVA, Akbank, İş Bankası ve Yapı Kredi'nin 2023, 2024 ve 2025 \
yılı entegre faaliyet raporları (Türkçe). Raporlar önceki yılın karşılaştırmalı rakamlarını \
(ör. 2023 raporunda 2022) ve bazı çok yıllık özet tabloları da içerir. Akbank, İş Bankası ve \
Yapı Kredi raporları tam finansal tabloları (konsolide ve solo) içerir; Garanti raporu içermez.

Görev: Bu korpustan CEVAPLANAMAYAN, ama bir kullanıcının makul olarak sorabileceği sorular \
yaz. Her türden {per_kind} soru:
- korpus_disi_banka: Korpusta olmayan bir Türk bankası hakkında (Ziraat Bankası, Halkbank, \
VakıfBank, QNB, Denizbank, TEB, ING, Şekerbank...). banks boş liste.
- yanlis_oncul: Korpustaki bir banka için yanlış bir öncül içeren soru (ör. kâr eden bir \
banka için "ne kadar zarar etti?", olmayan bir birleşmeyi, kapanmayı ya da olayı varsayan \
soru). Aşağıdaki doğrulanmış rakamlardan yararlan. banks: sorudaki banka(lar).
- olmayan_bilgi: Korpustaki bir banka için, faaliyet raporunda yer alması beklenmeyen ama \
makul görünen bir bilgi (ör. belirli bir şubenin müşteri sayısı, bir yöneticinin kişisel \
bilgisi, gelecek yıl hisse fiyatı hedefi, rakam bazında çok ince bir kırılım). banks: \
sorudaki banka(lar).
Kurallar: Soru kendi başına anlaşılır olsun; banka adını ve yılı (yıl gerekiyorsa) açıkça \
yazsın. Bankaları ve yılları (2023, 2024, 2025) dengeli dağıt. years: soruda geçen yıllar. \
why: neden cevaplanamadığı (kısa).

Doğrulanmış rakamlar (yanlış öncül için):
{fact_lines or "- (yok)"}"""


def validate_candidate(out: AbstainOut) -> None:
    """Raise ValueError when the candidate's fields contradict its text."""
    named = banks_mentioned(out.question)
    if out.kind == "korpus_disi_banka":
        if out.banks or named:
            raise ValueError("outside-corpus question names a corpus bank")
    elif not out.banks or named != set(out.banks):
        raise ValueError(f"banks {out.banks} != banks named in question {sorted(named)}")
    if set(out.years) != years_mentioned(out.question):
        raise ValueError(f"years {out.years} != years in question")
    if any(not YEAR_MIN <= y <= YEAR_MAX for y in out.years):
        raise ValueError(f"year outside {YEAR_MIN}-{YEAR_MAX}")


def doc_filter(out: AbstainOut, years: Sequence[int] = (2023, 2024, 2025)) -> list[str] | None:
    """Documents to search: all reports of the named banks, or the whole corpus."""
    if not out.banks:
        return None
    return [f"{b}-{y}" for b in out.banks for y in years]


def build_judge_prompt(items: Sequence[tuple[AbstainOut, Sequence[SearchHit]]]) -> str:
    blocks = []
    for i, (out, hits) in enumerate(items):
        passages = "\n\n".join(
            f"[P{j}] ({h.doc_id}, sayfa {h.page_start}-{h.page_end})\n{h.text}"
            for j, h in enumerate(hits, 1)
        )
        blocks.append(f"### Soru {i}: {out.question}\n\n{passages}")
    return (
        "Her soru için, altındaki pasajlar (korpustaki en ilgili pasajlar) temelinde karar ver:\n"
        "- answerable: pasajlar soruyu cevaplayan bilgiyi içeriyor.\n"
        "- partly_answerable: kısmen ya da yakın bir bilgi var; ya da emin değilsin.\n"
        "- false_premise: soru yanlış bir öncül içeriyor ve pasajlar öncülün yanlış olduğunu "
        "gösteriyor (ör. banka zarar değil kâr etmiş).\n"
        "- not_in_corpus: pasajlarda bu bilgi yok ve sorulan şey bu raporlarda yer almıyor "
        "gibi görünüyor.\n"
        "Unutma: raporlar önceki yılın karşılaştırmalı rakamlarını ve çok yıllık özetleri de "
        "içerir. evidence_passages: false_premise ise öncülü çürüten pasaj numaraları, "
        "answerable/partly ise cevabı içeren pasajlar; aksi halde boş liste. items listesinde "
        "her soru için index (soru numarası) ile bir kayıt döndür.\n\n" + "\n\n".join(blocks)
    )


def keep_reason(out: AbstainOut, verdict: AbstainVerdict | None) -> str | None:
    """None when the candidate is kept, else the drop reason."""
    if verdict is None:
        return "no judge verdict"
    wanted = "false_premise" if out.kind == "yanlis_oncul" else "not_in_corpus"
    if verdict.verdict != wanted:
        return f"judge: {verdict.verdict} (wanted {wanted}): {verdict.reason}"
    return None


def to_draft(
    key: str, out: AbstainOut, verdict: AbstainVerdict, hits: Sequence[SearchHit]
) -> Draft:
    evidence: list[Evidence] = []
    if out.kind == "yanlis_oncul":
        seen: set[tuple[str, int]] = set()
        for j in verdict.evidence_passages:
            if 1 <= j <= len(hits):
                h = hits[j - 1]
                for page in range(h.page_start, h.page_end + 1):
                    if (h.doc_id, page) not in seen:
                        seen.add((h.doc_id, page))
                        evidence.append(Evidence(doc=h.doc_id, page=page))
    label = {
        "korpus_disi_banka": "Korpus dışı banka",
        "yanlis_oncul": "Yanlış öncül",
        "olmayan_bilgi": "Raporlarda yer almayan bilgi",
    }[out.kind]
    return Draft(
        key=key,
        source="abstain",
        type="cevapsiz",
        question=out.question.strip(),
        banks=sorted(set(out.banks)),
        years=sorted(set(out.years)),
        basis=None,
        expected_behavior="abstain",
        expected=Expected(answer=f"{label}: {out.why}", value=None, unit=None, tolerance=None),
        evidence=evidence,
        llm_checked=True,
        notes=f"abstain kind: {out.kind}; corpus check: {verdict.verdict} - {verdict.reason}",
    )
