"""Dev/test split assignment (eval/README.md §2).

The split is stratified by question type and bank with a fixed seed (42), targets
60 dev : 120 test, and existing assignments never change. A bank stratum is the
question's single bank, `multi` for several banks and `none` for no bank.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from fractions import Fraction

from kaprag.eval.schema import Question, Split

SEED = 42
SPLIT_TARGETS: dict[Split, int] = {"dev": 60, "test": 120}


def stratum(question: Question) -> tuple[str, str]:
    """The (type, bank) stratum of a question."""
    if len(question.banks) == 1:
        return question.type, question.banks[0]
    return question.type, ("multi" if question.banks else "none")


def _tie_break(question_id: str) -> Split:
    """Deterministic pick weighted by the target shares, from a hash of id and seed."""
    digest = hashlib.sha256(f"{SEED}:{question_id}".encode()).digest()
    slot = int.from_bytes(digest[:8], "big") % sum(SPLIT_TARGETS.values())
    return "dev" if slot < SPLIT_TARGETS["dev"] else "test"


def assign_split_for_new(question: Question, existing: Iterable[Question]) -> Split:
    """Split for a question added after the initial split; `question.split` is ignored.

    Within the question's stratum, pick the split furthest below its target share
    (target count minus actual count, over the stratum's current size). An exact tie
    (stratum empty or exactly on ratio) is broken by `_tie_break(question.id)`.
    """
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
    return _tie_break(question.id)
