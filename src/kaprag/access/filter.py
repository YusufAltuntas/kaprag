"""The SQL access condition of a request (SIMULATED access model, `kaprag.access.roles`).

`access_filter(role, where)` ANDs `access_roles && <effective roles>` into a retrieval
filter. The roles are a bound parameter (`kaprag.retrieval.SqlFilter`), and the filter
goes into the WHERE clause of every retrieval arm (dense and full-text), so restricted
chunks are excluded before ranking, never removed from a ranked list afterwards.

The role comes from the configuration or the caller of the pipeline only, never from
the question text or from model output.
"""

from __future__ import annotations

from kaprag.access.roles import effective_roles
from kaprag.retrieval import SqlFilter

ACCESS_COLUMN = "access_roles"


def access_filter(role: str, where: SqlFilter | None = None) -> SqlFilter:
    """`where` (e.g. the bank/year metadata filter; None = no other condition) AND the
    access condition for `role`."""
    return (where or SqlFilter()).overlaps(ACCESS_COLUMN, list(effective_roles(role)))
