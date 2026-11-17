"""Split assignment for questions added after the initial split (eval/README.md §2)."""

import json
from pathlib import Path

from kaprag.eval.schema import Question
from kaprag.eval.split import assign_split_for_new, stratum

FIXTURE = Path(__file__).parent / "fixtures" / "eval" / "questions.jsonl"


def make(qid: str, type_: str = "tablo", banks=("akbank",), split: str = "dev") -> Question:
    data = json.loads(FIXTURE.read_text(encoding="utf-8").splitlines()[1])
    extra = {
        "cevapsiz": {"expected_behavior": "abstain"},
        "sentez": {"key_points": ["k"]},
    }.get(type_, {})
    return Question.model_validate(
        data | {"id": qid, "type": type_, "banks": list(banks), "split": split} | extra
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


# --- initial stratified split and pilot selection -------------------------------------


def _pool() -> list[Question]:
    """60 questions over a few strata of different sizes (sizes 30, 12, 9, 7, 2)."""
    spec = [("tablo", ("akbank",), 30), ("tablo", ("garanti",), 12),
            ("olgusal", ("isbank",), 9), ("karsilastirma", ("akbank", "isbank"), 7),
            ("cevapsiz", (), 2)]  # fmt: skip
    pool, n = [], 0
    for type_, banks, size in spec:
        for _ in range(size):
            n += 1
            pool.append(make(f"q{n:03d}", type_=type_, banks=banks, split="dev"))
    return pool


def test_initial_split_keeps_one_to_two_per_stratum():
    from collections import Counter

    from kaprag.eval.split import initial_split

    pool = _pool()
    splits = initial_split(pool)
    assert set(splits) == {q.id for q in pool}
    per_stratum: dict = {}
    for q in pool:
        per_stratum.setdefault(stratum(q), Counter())[splits[q.id]] += 1
    for key, counts in per_stratum.items():
        size = counts["dev"] + counts["test"]
        assert abs(counts["dev"] - size / 3) < 1, (key, counts)
    assert per_stratum[("tablo", "akbank")]["dev"] == 10


def test_initial_split_is_deterministic_and_order_independent():
    from kaprag.eval.split import initial_split

    pool = _pool()
    assert initial_split(pool) == initial_split(list(reversed(pool)))


def test_initial_split_depends_on_the_seed():
    from kaprag.eval.split import initial_split

    pool = _pool()
    assert initial_split(pool, seed=42) != initial_split(pool, seed=7)


def test_initial_split_agrees_with_assign_split_for_new():
    """Adding a question to the initial split is what assign_split_for_new would do: the
    initial split is the same rule applied one question at a time, so every stratum is
    left balanced in the sense assign_split_for_new uses."""
    from kaprag.eval.split import initial_split, split_order

    pool = _pool()
    splits = initial_split(pool)
    assigned: list[Question] = []
    for q in split_order(pool):
        assert assign_split_for_new(q, assigned) == splits[q.id]
        assigned.append(q.model_copy(update={"split": splits[q.id]}))


def test_pilot_is_30_from_dev_stratified_by_type():
    from collections import Counter

    from kaprag.eval.split import initial_split, select_pilot

    pool = []
    types = {"olgusal": 40, "tablo": 50, "hesaplama": 25, "sentez": 25}
    n = 0
    for type_, size in types.items():
        for i in range(size):
            n += 1
            bank = ("akbank", "garanti", "isbank", "yapikredi")[i % 4]
            pool.append(make(f"q{n:03d}", type_=type_, banks=(bank,)))
    splits = initial_split(pool)
    pool = [q.model_copy(update={"split": splits[q.id]}) for q in pool]
    pilot = select_pilot(pool, n=30)
    assert len(pilot) == 30
    by_id = {q.id: q for q in pool}
    assert all(by_id[i].split == "dev" for i in pilot)
    per_type = Counter(by_id[i].type for i in pilot)
    dev_per_type = Counter(q.type for q in pool if q.split == "dev")
    dev_total = sum(dev_per_type.values())
    for type_, count in dev_per_type.items():
        assert abs(per_type[type_] - 30 * count / dev_total) <= 1, (type_, per_type)
    assert select_pilot(list(reversed(pool)), n=30) == pilot


def test_pilot_takes_all_dev_when_dev_is_small():
    from kaprag.eval.split import select_pilot

    pool = [make("a", split="dev"), make("b", split="test")]
    assert select_pilot(pool, n=30) == ["a"]
