"""Dev/test split assignment and pilot selection (eval/README.md §2).

The split is stratified by question type and bank with a fixed seed (42), targets
60 dev : 120 test, and existing assignments never change. A bank stratum is the
question's single bank, `multi` for several banks and `none` for no bank.

`initial_split` is the same rule as `assign_split_for_new`, applied one question at
a time in a seeded order, so the two can never disagree about a balanced stratum.
`select_pilot` draws the pilot questions (§2, §7 step 9) from the dev split only.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterable, Sequence
from fractions import Fraction

from kaprag.eval.schema import Question, Split

SEED = 42
SPLIT_TARGETS: dict[Split, int] = {"dev": 60, "test": 120}


def stratum(question: Question) -> tuple[str, str]:
    """The (type, bank) stratum of a question."""
    if len(question.banks) == 1:
        return question.type, question.banks[0]
    return question.type, ("multi" if question.banks else "none")


def _tie_break(question_id: str, seed: int = SEED) -> Split:
    """Deterministic pick weighted by the target shares, from a hash of id and seed."""
    digest = hashlib.sha256(f"{seed}:{question_id}".encode()).digest()
    slot = int.from_bytes(digest[:8], "big") % sum(SPLIT_TARGETS.values())
    return "dev" if slot < SPLIT_TARGETS["dev"] else "test"


def assign_split_for_new(question: Question, existing: Iterable[Question]) -> Split:
    """Split for a question added after the initial split; `question.split` is ignored.

    Within the question's stratum, pick the split furthest below its target share
    (target count minus actual count, over the stratum's current size). An exact tie
    (stratum empty or exactly on ratio) is broken by `_tie_break(question.id)`.
    """
    return _assign(question, existing, SEED)


def _assign(question: Question, existing: Iterable[Question], seed: int) -> Split:
    key = stratum(question)
    counts: dict[Split, int] = dict.fromkeys(SPLIT_TARGETS, 0)
    for other in existing:
        if other.id != question.id and stratum(other) == key:
            counts[other.split] += 1
    size = sum(counts.values())
    total = sum(SPLIT_TARGETS.values())
    deficit = {s: Fraction(SPLIT_TARGETS[s] * size, total) - counts[s] for s in SPLIT_TARGETS}
    best = max(deficit.values())
    candidates = [s for s in SPLIT_TARGETS if deficit[s] == best]
    if len(candidates) == 1:
        return candidates[0]
    return _tie_break(question.id, seed)


def _seeded_key(question_id: str, seed: int, purpose: str) -> bytes:
    return hashlib.sha256(f"{seed}:{purpose}:{question_id}".encode()).digest()


def split_order(questions: Iterable[Question], seed: int = SEED) -> list[Question]:
    """The seeded order in which `initial_split` assigns questions (input order ignored)."""
    return sorted(questions, key=lambda q: (_seeded_key(q.id, seed, "split"), q.id))


def initial_split(questions: Sequence[Question], seed: int = SEED) -> dict[str, Split]:
    """Stratified dev/test split of a fresh set: question id -> split.

    Questions are taken in `split_order` and each is assigned with the
    `assign_split_for_new` rule against the ones assigned before it, so within every
    (type, bank) stratum dev stays within one question of a third (60:120). The
    input's own `split` values are ignored.
    """
    splits: dict[str, Split] = {}
    by_stratum: dict[tuple[str, str], list[Question]] = defaultdict(list)
    for question in split_order(questions, seed):
        key = stratum(question)
        split = _assign(question, by_stratum[key], seed)
        splits[question.id] = split
        by_stratum[key].append(question.model_copy(update={"split": split}))
    return splits


def select_pilot(questions: Sequence[Question], n: int = 30, seed: int = SEED) -> list[str]:
    """Ids of `n` pilot questions from the dev split, stratified by type (sorted ids).

    Seats are shared out over types in proportion to their dev counts (largest
    remainder, ties by type name); within a type, questions are taken round-robin over
    bank strata in a seeded order, so one bank does not fill a type's seats.
    """
    dev = [q for q in questions if q.split == "dev"]
    if len(dev) <= n:
        return sorted(q.id for q in dev)
    by_type: dict[str, list[Question]] = defaultdict(list)
    for q in dev:
        by_type[q.type].append(q)
    quotas = {t: Fraction(n * len(qs), len(dev)) for t, qs in by_type.items()}
    seats = {t: int(quota) for t, quota in quotas.items()}
    by_remainder = sorted(quotas, key=lambda t: (-(quotas[t] - seats[t]), t))
    for t in by_remainder[: n - sum(seats.values())]:
        seats[t] += 1
    chosen: list[str] = []
    for t, qs in by_type.items():
        chosen += _round_robin_by_bank(qs, seed)[: seats[t]]
    return sorted(chosen)


def _round_robin_by_bank(questions: list[Question], seed: int) -> list[str]:
    groups: dict[str, list[Question]] = defaultdict(list)
    for q in questions:
        groups[stratum(q)[1]].append(q)
    queues = [
        sorted(groups[bank], key=lambda q: (_seeded_key(q.id, seed, "pilot"), q.id))
        for bank in sorted(groups)
    ]
    order: list[str] = []
    while any(queues):
        for queue in queues:
            if queue:
                order.append(queue.pop(0).id)
    return order
