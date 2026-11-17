"""Leak counting of the access-control leak test, with fake retrieval results."""

import json

import pytest

from kaprag.access.leak import (
    DEFAULT_LEAK_QUERIES,
    LeakQuery,
    check_query,
    load_leak_queries,
    summarize,
    verdict,
)

ROLES_OF = {
    "a:fixed:1": ["public"],
    "a:fixed:2": ["analyst", "risk"],
    "a:fixed:3": ["risk"],
    "a:fixed:4": ["public"],
}
Q = LeakQuery(id="acl01", query="soru", attack="direct", target="risk yönetimi")


def test_no_leak_when_every_item_is_visible_to_the_role():
    result = check_query(Q, ["a:fixed:1", "a:fixed:4"], ROLES_OF, "public")
    assert result.leaked == []
    assert result.retrieved == ["a:fixed:1", "a:fixed:4"]


def test_leak_detected_anywhere_in_the_full_ranked_list():
    ranked = ["a:fixed:1", "a:fixed:4"] + [f"a:fixed:{i}" for i in (1, 4) * 10] + ["a:fixed:3"]
    result = check_query(Q, ranked, ROLES_OF, "public")
    assert result.leaked == ["a:fixed:3"]  # the last of 23 items, far below a top-5 context


def test_restricted_items_per_role():
    ranked = ["a:fixed:1", "a:fixed:2", "a:fixed:3"]
    assert check_query(Q, ranked, ROLES_OF, "public").leaked == ["a:fixed:2", "a:fixed:3"]
    assert check_query(Q, ranked, ROLES_OF, "analyst").leaked == ["a:fixed:3"]
    assert check_query(Q, ranked, ROLES_OF, "risk").leaked == []


def test_unknown_chunk_fails_loudly():
    with pytest.raises(KeyError, match="a:fixed:9"):
        check_query(Q, ["a:fixed:9"], ROLES_OF, "public")


def test_summary_counts_leaking_queries_and_chunks():
    results = [
        check_query(Q, ["a:fixed:1", "a:fixed:3"], ROLES_OF, "public"),
        check_query(Q.model_copy(update={"id": "acl02"}), ["a:fixed:4"], ROLES_OF, "public"),
        check_query(
            Q.model_copy(update={"id": "acl03"}), ["a:fixed:2", "a:fixed:3"], ROLES_OF, "public"
        ),
    ]
    summary = summarize(results)
    assert summary == {
        "queries": 3,
        "queries_with_leaks": 2,
        "leaked_items": 3,
        "leaking_query_ids": ["acl01", "acl03"],
        "leaked_chunk_ids": ["a:fixed:2", "a:fixed:3"],
    }


def _result(qid, ranked, role="public"):
    return check_query(Q.model_copy(update={"id": qid}), ranked, ROLES_OF, role)


CONTROL = [_result("acl01", ["a:fixed:1", "a:fixed:3"]), _result("acl02", ["a:fixed:4"])]


def test_verdict_passes_with_no_leaks_full_lists_and_a_positive_control():
    results = [_result("acl01", ["a:fixed:1"]), _result("acl02", ["a:fixed:4"])]
    assert verdict(results, CONTROL) == (0, None)


def test_verdict_fails_on_a_leak():
    results = [_result("acl01", ["a:fixed:1", "a:fixed:3"]), _result("acl02", [])]
    code, reason = verdict(results, CONTROL)
    assert code == 1 and "leak" in reason


def test_verdict_refuses_an_empty_filtered_list():
    # 0 leaks, but the filter starved a query's list to nothing: the pass would be vacuous
    results = [_result("acl01", ["a:fixed:1"]), _result("acl02", [])]
    code, reason = verdict(results, CONTROL)
    assert code == 2 and "acl02" in reason


def test_verdict_refuses_a_positive_control_without_restricted_chunks():
    results = [_result("acl01", ["a:fixed:1"]), _result("acl02", ["a:fixed:4"])]
    control = [_result("acl01", ["a:fixed:1"]), _result("acl02", ["a:fixed:4"])]
    code, reason = verdict(results, control)
    assert code == 2 and "positive control" in reason


def test_load_leak_queries_validates(tmp_dir):
    path = tmp_dir / "q.jsonl"
    rows = [Q.model_dump(), {**Q.model_dump(), "id": "acl02", "attack": "injection"}]
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    assert [q.id for q in load_leak_queries(path)] == ["acl01", "acl02"]

    path.write_text("\n".join(json.dumps(r) for r in [rows[0], rows[0]]), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_leak_queries(path)

    path.write_text(json.dumps({**rows[0], "attack": "other"}), encoding="utf-8")
    with pytest.raises(ValueError, match="attack"):
        load_leak_queries(path)


def test_shipped_leak_queries():
    queries = load_leak_queries(DEFAULT_LEAK_QUERIES)
    assert len(queries) == 30
    kinds = {q.attack for q in queries}
    assert kinds == {"direct", "injection"}
    assert sum(q.attack == "injection" for q in queries) >= 10
    assert any("risk ekibindenim" in q.query.casefold() for q in queries)
