"""The SIMULATED role model: roles, a request's effective roles, a chunk's role set.

The annual reports are public documents; these roles are assigned by rule
(`kaprag.access.rules`) purely to demonstrate access filtering.

A request carries ONE role. Its effective roles always include `public`:

    public  -> {public}
    analyst -> {public, analyst}
    risk    -> {public, risk}

A chunk's `access_roles` lists the roles that may retrieve it. There are three sets:

    {public}          everything else
    {analyst, risk}   financial-statement footnotes
    {risk}            risk-management content

A chunk is visible when `access_roles && effective_roles` (non-empty intersection), so
`risk` sees everything, `analyst` sees public content and footnotes, `public` sees
only public content. When a chunk qualifies for several sets (e.g. a fixed chunk
spanning a public page and a risk page), the most restrictive one wins, in the order
{public} < {analyst, risk} < {risk}: {risk} is visible to fewer roles, so a chunk
with both risk and footnote content is {risk}. Over-restriction is preferred to a leak.
"""

from collections.abc import Iterable

ROLES: tuple[str, ...] = ("public", "analyst", "risk")

PUBLIC = frozenset({"public"})
FOOTNOTES = frozenset({"analyst", "risk"})
RISK = frozenset({"risk"})

# The role sets a chunk can carry, least to most restrictive.
CHUNK_ROLE_SETS: tuple[frozenset[str], ...] = (PUBLIC, FOOTNOTES, RISK)


def effective_roles(role: str) -> tuple[str, ...]:
    """The roles a request with `role` acts as: `public` plus the role itself."""
    if role not in ROLES:
        raise ValueError(f"unknown role {role!r}; expected one of {ROLES}")
    return tuple(r for r in ROLES if r in {"public", role})


def can_see(chunk_roles: Iterable[str], role: str) -> bool:
    """Whether a request with `role` may retrieve a chunk carrying `chunk_roles`."""
    return not set(chunk_roles).isdisjoint(effective_roles(role))


def most_restrictive(role_sets: Iterable[frozenset[str]]) -> frozenset[str]:
    """The most restrictive of chunk role sets; {public} for none."""
    rank = 0
    for roles in role_sets:
        if roles not in CHUNK_ROLE_SETS:
            raise ValueError(f"{sorted(roles)} is not a chunk role set")
        rank = max(rank, CHUNK_ROLE_SETS.index(roles))
    return CHUNK_ROLE_SETS[rank]


def role_list(roles: Iterable[str]) -> list[str]:
    """Roles as a list in vocabulary order (the stored `access_roles` value)."""
    present = set(roles)
    return [r for r in ROLES if r in present]
