"""Answer generation: system prompt, the model's JSON schema, and the LLM call.

The model sees the question and the numbered sources (`kaprag.rag.context`) and
replies with `GeneratedAnswer`. It reports a figure as printed in the source plus its
unit including any scale word ("45,2" + "milyar TL"); converting to canonical units and
mapping source numbers to pages is done in code (`kaprag.rag.postprocess`).

This is the baseline prompt: behaviour rules only, no few-shot examples, query
rewriting or self-check passes, so later ablations are measured against it.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from kaprag.llm import LLMClient

SYSTEM_PROMPT = """\
Sen Garanti BBVA, Akbank, İş Bankası ve Yapı Kredi'nin 2023-2025 faaliyet raporları \
üzerinde çalışan bir finansal analiz asistanısın. Soruyu YALNIZCA verilen numaralı \
kaynaklara dayanarak cevapla; kendi bilgini kullanma.

Kurallar:
1. Cevabı sorunun dilinde yaz. Dayandığın kaynağı cümle içinde [n] ile göster ve \
cevabı destekleyen kaynak numaralarını `citations` alanına yaz.
2. Kaynaklar cevabı içermiyorsa (banka, yıl ya da metrik kaynaklarda yok, ya da \
sorunun öncülü kaynaklarla çelişiyor) tahmin etme: `abstained: true` yap, `answer` \
alanında kısaca nedenini söyle, `value` ve `unit` boş kalsın.
3. Soru hangi bankayı ya da hangi yılı kastettiğini söylemiyor ve bu yüzden cevap \
belirlenemiyorsa tahmin etme: `clarification` alanına kullanıcıya sorulacak \
netleştirme sorusunu yaz (`abstained: false`).
4. Soru konsolide mi solo (bireysel) mu belirtmiyorsa ve kaynaklarda konsolide rakam \
varsa konsolide rakamı kullan ve cevapta bunu varsaydığını açıkça söyle. Finansal bir \
rakam verdiğinde, kaynaktan anlaşılıyorsa `basis` alanını `konsolide` ya da `solo` yap.
5. Sayısal cevapta `value` alanına sayıyı kaynakta yazıldığı gibi yaz (ör. "45,2" ya \
da "1.234.567"); ölçek çarpanını kendin uygulama. `unit` alanına birimi, tablonun \
ölçek ifadesiyle birlikte yaz (ör. "milyar TL", "bin TL", "milyon TL", "%", "adet"). \
Hesaplama gerekiyorsa (ör. yüzde değişim) sonucu `value`, birimini `unit` alanına \
yaz. Sayısal olmayan sorularda `value` ve `unit` boş kalır."""


class GeneratedAnswer(BaseModel):
    """The model's reply. Not the section 5 answer: see `kaprag.rag.postprocess`."""

    answer: str = Field(description="Sorunun dilinde cevap; kaynaklara [n] ile atıf.")
    value: str | float | int | None = Field(
        default=None,
        description='Sayısal cevapta sayı, kaynakta yazıldığı gibi (ör. "45,2"); yoksa null.',
    )
    unit: str | None = Field(
        default=None,
        description='Ölçek ifadesiyle birim (ör. "milyar TL", "bin TL", "%", "adet").',
    )
    basis: Literal["konsolide", "solo"] | None = Field(
        default=None, description="Finansal rakamın esası; belirlenemiyorsa null."
    )
    citations: list[int] = Field(
        default_factory=list, description="Cevabı destekleyen kaynak numaraları."
    )
    abstained: bool = Field(default=False, description="Kaynaklar cevabı içermiyorsa true.")
    clarification: str | None = Field(
        default=None, description="Banka/yıl belirsizse kullanıcıya sorulacak soru; yoksa null."
    )


def build_user_prompt(question: str, context: str) -> str:
    return f"Soru: {question}\n\nKaynaklar:\n\n{context}"


async def generate(
    client: LLMClient,
    question: str,
    context: str,
    *,
    model: str,
    tags: dict[str, Any],
) -> GeneratedAnswer:
    """One schema-validated LLM call. `LLMCallError` propagates to the caller."""
    result = await client.complete(
        SYSTEM_PROMPT,
        build_user_prompt(question, context),
        model,
        schema=GeneratedAnswer,
        tags={**tags, "step": "generate"},
    )
    return result.parsed
