"""Step 6: `belirsiz` drafts, derived from checked singles by removing one piece of info.

- `drop_bank` / `drop_year` -> expected behaviour `clarify` (no expected value).
- `drop_basis` (source figure consolidated) -> `answer_with_assumption`: the system
  should answer with the consolidated figure and say so (eval/README.md §4); the expected
  value and evidence are the source's, `basis: konsolide`.

One LLM call rewrites the selected source questions; code checks that the removed
element is really gone and the rest is still there. `derived_from` points to the source.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from pydantic import BaseModel

from kaprag.eval.schema import Expected
from kaprag.gold.draft import Draft
from kaprag.gold.text import banks_mentioned, bases_mentioned, mentions_bank, years_mentioned

KINDS = ("drop_bank", "drop_year", "drop_basis")


class Rewrite(BaseModel):
    index: int
    question: str


class Rewrites(BaseModel):
    items: list[Rewrite]


@dataclass(frozen=True)
class AmbiguousTask:
    kind: str
    source: Draft


SYSTEM = (
    "Sen soru metinlerini verilen talimata göre en az değişiklikle yeniden yazan bir editörsün."
)

_HOW = {
    "drop_bank": "Banka adını sorudan çıkar (başka banka adı da yazma); yıl ve diğer her şey "
    "kalsın.",
    "drop_year": "Yılı ve yıl ifadesini sorudan çıkar (başka yıl yazma); banka ve diğer her şey "
    "kalsın.",
    "drop_basis": "'konsolide' / 'solo' / 'konsolide bazda' gibi esas ifadesini sorudan çıkar; "
    "banka, yıl ve diğer her şey kalsın.",
}


def _hash(seed: int, key: str) -> bytes:
    return hashlib.sha256(f"{seed}:ambiguous:{key}".encode()).digest()


def select_sources(singles: Sequence[Draft], per_kind: int, seed: int = 42) -> list[AmbiguousTask]:
    """Deterministic, bank-balanced choice of sources: numeric checked singles of one bank
    and one year; `drop_basis` needs a consolidated source. A source is used once."""
    pool = [
        d
        for d in singles
        if d.llm_checked
        and d.expected.value is not None
        and len(d.banks) == 1
        and len(d.years) == 1
        and d.type in ("tablo", "olgusal")
    ]
    used: set[str] = set()
    tasks: list[AmbiguousTask] = []
    for kind in ("drop_basis", "drop_bank", "drop_year"):
        eligible = [d for d in pool if kind != "drop_basis" or d.basis == "konsolide"]
        by_bank: dict[str, list[Draft]] = defaultdict(list)
        for d in sorted(eligible, key=lambda d: (d.source != "probe", _hash(seed, kind + d.key))):
            by_bank[d.banks[0]].append(d)
        queues = [by_bank[b] for b in sorted(by_bank)]
        taken = 0
        while taken < per_kind and any(queues):
            for queue in queues:
                while queue and queue[0].key in used:
                    queue.pop(0)
                if queue and taken < per_kind:
                    d = queue.pop(0)
                    used.add(d.key)
                    tasks.append(AmbiguousTask(kind, d))
                    taken += 1
    return tasks


def build_prompt(tasks: Sequence[AmbiguousTask]) -> str:
    lines = [
        "Her soruyu yanındaki talimata göre yeniden yaz. Dil doğal kalsın; anlamı başka hiçbir "
        "yönden değiştirme. items listesinde her soru için index ve yeni soru döndür.",
        "",
    ]
    for i, t in enumerate(tasks):
        lines.append(f"[{i}] Talimat: {_HOW[t.kind]}\nSoru: {t.source.question}")
    return "\n".join(lines)


def to_draft(task: AmbiguousTask, question: str) -> Draft:
    """The belirsiz draft for a rewritten question; ValueError if the rewrite is wrong."""
    src = task.source
    bank, year = src.banks[0], src.years[0]
    question = question.strip()
    if question == src.question:
        raise ValueError("rewrite is unchanged")
    names_bank = mentions_bank(question, bank)
    years = years_mentioned(question)
    if task.kind == "drop_bank":
        if banks_mentioned(question):
            raise ValueError("a bank is still named")
        if years != {year}:
            raise ValueError("year changed or missing")
    elif task.kind == "drop_year":
        if years:
            raise ValueError("a year is still named")
        if not names_bank:
            raise ValueError("bank missing")
    else:
        if bases_mentioned(question):
            raise ValueError("basis still named")
        if not names_bank or years != {year}:
            raise ValueError("bank or year missing")
    base = {
        "key": f"ambiguous:{task.kind}:{src.key}",
        "source": "ambiguous",
        "type": "belirsiz",
        "question": question,
        "derived_from": [src.key],
        # clarify drafts have no answer a check could confirm; drop_basis keeps the
        # checked source answer
        "llm_checked": src.llm_checked and task.kind == "drop_basis",
    }
    if task.kind == "drop_basis":
        answer = f"Konsolide rakam varsayılarak: {src.expected.answer}"
        return Draft(
            **base,
            banks=[bank],
            years=[year],
            basis="konsolide",
            expected_behavior="answer_with_assumption",
            expected=src.expected.model_copy(update={"answer": answer}),
            evidence=list(src.evidence),
            notes="basis removed from a consolidated question; answer with consolidated figure "
            "and state the assumption",
        )
    missing = "banka" if task.kind == "drop_bank" else "yıl"
    return Draft(
        **base,
        banks=[] if task.kind == "drop_bank" else [bank],
        years=[year] if task.kind == "drop_bank" else [],
        basis=None,
        expected_behavior="clarify",
        expected=Expected(
            answer=f"Netleştirme gerekli: soru hangi {missing} olduğunu belirtmiyor.",
            value=None,
            unit=None,
            tolerance=None,
        ),
        evidence=[],
        notes=f"{missing} removed from the source question",
    )
