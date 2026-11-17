"""The simulated role model: vocabulary, effective roles, chunk role sets, precedence."""

import pytest

from kaprag.access.roles import (
    CHUNK_ROLE_SETS,
    FOOTNOTES,
    PUBLIC,
    RISK,
    ROLES,
    can_see,
    effective_roles,
    most_restrictive,
    role_list,
)


def test_vocabulary():
    assert ROLES == ("public", "analyst", "risk")
    assert frozenset({"public"}) == PUBLIC
    assert frozenset({"analyst", "risk"}) == FOOTNOTES
    assert frozenset({"risk"}) == RISK
    assert CHUNK_ROLE_SETS == (PUBLIC, FOOTNOTES, RISK)  # least to most restrictive


@pytest.mark.parametrize(
    ("role", "expected"),
    [
        ("public", ("public",)),
        ("analyst", ("public", "analyst")),
        ("risk", ("public", "risk")),
    ],
)
def test_effective_roles_always_include_public(role, expected):
    assert effective_roles(role) == expected


@pytest.mark.parametrize("role", ["admin", "Risk", "", "public,risk", None])
def test_unknown_role_is_refused(role):
    with pytest.raises(ValueError, match="unknown role"):
        effective_roles(role)


@pytest.mark.parametrize(
    ("role", "visible"),
    [
        ("public", {"public"}),
        ("analyst", {"public", "footnotes"}),
        ("risk", {"public", "footnotes", "risk"}),
    ],
)
def test_who_sees_what(role, visible):
    sets = {"public": PUBLIC, "footnotes": FOOTNOTES, "risk": RISK}
    assert {name for name, roles in sets.items() if can_see(roles, role)} == visible


def test_most_restrictive_precedence():
    assert most_restrictive([]) == PUBLIC
    assert most_restrictive([PUBLIC, PUBLIC]) == PUBLIC
    assert most_restrictive([PUBLIC, FOOTNOTES]) == FOOTNOTES
    # {risk} is visible to fewer roles than {analyst, risk}: it wins when both occur
    assert most_restrictive([FOOTNOTES, RISK, PUBLIC]) == RISK
    assert most_restrictive([RISK, FOOTNOTES]) == RISK


def test_most_restrictive_refuses_unknown_sets():
    with pytest.raises(ValueError, match="not a chunk role set"):
        most_restrictive([frozenset({"analyst"})])


def test_role_list_is_in_vocabulary_order():
    assert role_list(FOOTNOTES) == ["analyst", "risk"]
    assert role_list(frozenset({"risk", "public"})) == ["public", "risk"]
