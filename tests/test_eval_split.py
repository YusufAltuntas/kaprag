"""Split assignment for questions added after the initial split (eval/README.md §2)."""

import json
from pathlib import Path

from kaprag.eval.schema import Question
from kaprag.eval.split import assign_split_for_new, stratum

FIXTURE = Path(__file__).parent / "fixtures" / "eval" / "questions.jsonl"


def make(qid: str, type_: str = "tablo", banks=("akbank",), split: str = "dev") -> Question:
    data = json.loads(FIXTURE.read_text(encoding="utf-8").splitlines()[1])
    return Question.model_validate(
        data | {"id": qid, "type": type_, "banks": list(banks), "split": split}
    )


def test_stratum_is_type_and_bank():
    assert stratum(make("a")) == ("tablo", "akbank")
    assert stratum(make("a", banks=("akbank", "garanti"))) == ("tablo", "multi")
    assert stratum(make("a", banks=())) == ("tablo", "none")


def test_picks_split_below_its_share():
    # dev:test target is 60:120 -> one dev per two test
    two_dev = [make("a", split="dev"), make("b", split="dev")]
    assert assign_split_for_new(make("q1"), two_dev) == "test"
    two_test = [make("a", split="test"), make("b", split="test")]
    assert assign_split_for_new(make("q1"), two_test) == "dev"


def test_only_the_same_stratum_counts():
    others = [make(f"o{i}", banks=("garanti",), split="test") for i in range(9)]
    others += [make(f"t{i}", type_="olgusal", split="test") for i in range(9)]
    one_test = [make("a", split="test")]
    assert assign_split_for_new(make("q1"), others + one_test) == "dev"


def test_exact_tie_is_deterministic_hash_of_id():
    balanced = [make("a", split="dev"), make("b", split="test"), make("c", split="test")]
    for existing in ([], balanced):
        picks = {qid: assign_split_for_new(make(qid), existing) for qid in
                 [f"q{i:03d}" for i in range(1, 61)]}  # fmt: skip
        again = {qid: assign_split_for_new(make(qid), existing) for qid in picks}
        assert picks == again
        assert set(picks.values()) == {"dev", "test"}  # the hash really varies


def test_ignores_the_new_questions_own_split():
    existing = [make("a", split="dev")]
    assert assign_split_for_new(make("q1", split="dev"), existing) == "test"
    assert assign_split_for_new(make("q1", split="test"), existing) == "test"


def test_adding_many_keeps_one_to_two_ratio():
    existing: list[Question] = []
    for i in range(1, 31):
        q = make(f"q{i:03d}")
        existing.append(q.model_copy(update={"split": assign_split_for_new(q, existing)}))
    dev = sum(q.split == "dev" for q in existing)
    assert dev == 10
