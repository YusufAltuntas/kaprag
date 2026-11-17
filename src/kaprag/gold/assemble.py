"""Final assembly: per-type quotas, ids, dev/test split and pilot flags.

Selection order: every type except `tablo`/`olgusal` first (derived and page drafts
compete in one pool per type); the singles they reference (`derived_from`) are then
forced in, because a derived question is only verifiable through its sources; the
remaining `tablo`/`olgusal` quota is filled from the other checked drafts. Within a
type, drafts are taken round-robin over cells - (bank, year) for single-bank drafts,
(metric, year) for multi-bank ones, the kind for cevapsiz - in a seeded order, so no
bank, year, metric or kind dominates. Ids `q001...` follow a stable sort (type order, banks, years,
key); then `kaprag.eval.split.initial_split` and `select_pilot` run on the result.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Mapping, Sequence

from kaprag.eval.schema import QUESTION_TYPES, Question
from kaprag.eval.split import SEED, initial_split, select_pilot
from kaprag.gold.draft import Draft, to_question

# eval/README.md §1 targets, and the default drafting quotas (~1.3x, room for rejects).
SPEC_TARGETS: dict[str, int] = {
    "olgusal": 30,
    "tablo": 40,
    "hesaplama": 20,
    "karsilastirma": 30,
    "sentez": 20,
    "cevapsiz": 25,
    "belirsiz": 15,
}
DEFAULT_QUOTAS: dict[str, int] = {
    "olgusal": 39,
    "tablo": 60,  # 1.5x: probe facts (headline metrics) take many seats; room for page tables
    "hesaplama": 26,
    "karsilastirma": 39,
    "sentez": 26,
    "cevapsiz": 33,
    "belirsiz": 20,
}
PILOT_SIZE = 30


def _cell(d: Draft) -> tuple:
    if d.source == "abstain":  # balance the three cevapsiz kinds
        return ("abstain", d.metric or "")
    if len(d.banks) == 1:
        return ("bank", d.banks[0], d.years[0] if d.years else 0)
    return ("multi", d.metric or "", tuple(d.years))


def balanced_pick(drafts: Sequence[Draft], n: int, seed: int = SEED) -> list[Draft]:
    """Up to `n` drafts, round-robin over cells, each cell in seeded order."""
    cells: dict[tuple, list[Draft]] = defaultdict(list)
    for d in drafts:
        cells[_cell(d)].append(d)
    queues = [
        sorted(cells[c], key=lambda d: (hashlib.sha256(f"{seed}:{d.key}".encode()).digest(), d.key))
        for c in sorted(cells, key=str)
    ]
    out: list[Draft] = []
    while len(out) < n and any(queues):
        for queue in queues:
            if queue and len(out) < n:
                out.append(queue.pop(0))
    return out


# Types whose drafts may be forced in as sources of other drafts; they are filled last.
SOURCE_TYPES = ("tablo", "olgusal")


def select(pool: Sequence[Draft], quotas: Mapping[str, int], seed: int = SEED) -> list[Draft]:
    """The drafts to publish, honouring quotas and forcing in referenced sources.

    Every type except the source types is picked first, from ALL its drafts (so page-YoY
    `hesaplama` competes with probe-derived gaps); the sources those drafts reference are
    then forced in, and `tablo` / `olgusal` are filled up to their quotas last.
    """
    by_key = {d.key: d for d in pool}
    if len(by_key) != len(pool):
        raise ValueError("duplicate draft keys")
    chosen: dict[str, Draft] = {}
    for type_ in QUESTION_TYPES:
        if type_ in SOURCE_TYPES:
            continue
        candidates = [
            d for d in pool if d.type == type_ and all(k in by_key for k in d.derived_from)
        ]
        for d in balanced_pick(candidates, quotas.get(type_, 0), seed):
            chosen[d.key] = d
    for d in list(chosen.values()):
        for k in d.derived_from:
            chosen.setdefault(k, by_key[k])
    for type_ in SOURCE_TYPES:
        have = sum(d.type == type_ for d in chosen.values())
        rest = [d for d in pool if d.type == type_ and d.key not in chosen]
        for d in balanced_pick(rest, quotas.get(type_, 0) - have, seed):
            chosen[d.key] = d
    return list(chosen.values())


def drop_probe_duplicates(
    page_singles: Sequence[Draft], probes: Sequence[Draft]
) -> tuple[list[Draft], list[tuple[Draft, str]]]:
    """Remove page singles that state the same figure as a probe: same evidence page(s),
    same normalised value and unit. The probe is kept because derived questions point at
    it. Returns (kept, [(dropped draft, probe key)])."""

    def fact(d: Draft) -> tuple:
        pages = tuple(sorted((e.doc, e.page or 0) for e in d.evidence))
        return pages, d.expected.value, d.expected.unit

    by_fact = {fact(p): p.key for p in probes if p.expected.value is not None}
    kept: list[Draft] = []
    dropped: list[tuple[Draft, str]] = []
    for d in page_singles:
        key = by_fact.get(fact(d)) if d.expected.value is not None else None
        if key is not None and d.type in ("tablo", "olgusal"):
            dropped.append((d, key))
        else:
            kept.append(d)
    return kept, dropped


def _sort_key(d: Draft) -> tuple:
    return (QUESTION_TYPES.index(d.type), d.banks, d.years, d.key)


def assemble(
    pool: Sequence[Draft],
    quotas: Mapping[str, int] = DEFAULT_QUOTAS,
    pilot_size: int = PILOT_SIZE,
    seed: int = SEED,
) -> list[Question]:
    """Questions for the gold file: selected, numbered, split and pilot-flagged."""
    return finalize(select(pool, quotas, seed), pilot_size, seed)


def finalize(
    selected: Sequence[Draft], pilot_size: int = PILOT_SIZE, seed: int = SEED
) -> list[Question]:
    """Number the selected drafts (stable order), then split and flag the pilot."""
    drafts = sorted(selected, key=_sort_key)
    width = max(3, len(str(len(drafts))))
    key_to_id = {d.key: f"q{i:0{width}d}" for i, d in enumerate(drafts, start=1)}
    questions = [
        to_question(d, qid=key_to_id[d.key], split="dev", pilot=False, key_to_id=key_to_id)
        for d in drafts
    ]
    splits = initial_split(questions, seed)
    questions = [q.model_copy(update={"split": splits[q.id]}) for q in questions]
    pilot = set(select_pilot(questions, pilot_size, seed))
    return [q.model_copy(update={"pilot": q.id in pilot}) for q in questions]
