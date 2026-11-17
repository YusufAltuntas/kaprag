"""Recompute and apply `access_roles` for rows already in a chunk table, without
re-chunking or re-embedding (SIMULATED access model, `kaprag.access`).

A row's role set follows from what is stored with it (`section_path`, `chunk_type`,
page span) and the document's per-page role map (`kaprag.access.rules`), exactly as at
indexing time (`kaprag.index.store.access_roles_for`). `apply_roles` updates only the
`access_roles` column, only of rows whose value changes, in one transaction per
document, so rerunning it is a no-op.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from psycopg import Connection, sql
from psycopg.pq import TransactionStatus

from kaprag.access.roles import role_list
from kaprag.access.rules import DocumentAccess, chunk_roles
from kaprag.index.store import table_name


@dataclass(frozen=True)
class RowRoles:
    chunk_id: str
    before: tuple[str, ...]
    after: tuple[str, ...]


def document_roles(
    conn: Connection, variant: str, doc_id: str, access: DocumentAccess
) -> list[RowRoles]:
    """Current and recomputed roles of every row of `doc_id`, in ordinal order."""
    rows = conn.execute(
        sql.SQL(
            "SELECT chunk_id, section_path, chunk_type, page_start, page_end, access_roles"
            " FROM {} WHERE doc_id = %s ORDER BY ordinal"
        ).format(sql.Identifier(table_name(variant))),
        (doc_id,),
    ).fetchall()
    return [
        RowRoles(
            chunk_id,
            tuple(current),
            tuple(role_list(chunk_roles(variant, path, ctype, start, end, access))),
        )
        for chunk_id, path, ctype, start, end, current in rows
    ]


def apply_roles(conn: Connection, variant: str, doc_id: str, rows: Sequence[RowRoles]) -> int:
    """Write `after` into `access_roles` for the rows where it differs, in one committed
    transaction; returns the number of rows updated. Touches no other column."""
    status = conn.info.transaction_status
    if status == TransactionStatus.INTRANS:
        conn.commit()  # the write is its own top-level transaction (see replace_document)
    elif status != TransactionStatus.IDLE:
        raise RuntimeError(f"connection not idle ({status.name}); roll back first")
    changed = [r for r in rows if r.after != r.before]
    if not changed:
        return 0
    query = sql.SQL(
        "UPDATE {} SET access_roles = %s WHERE chunk_id = %s AND doc_id = %s"
        " AND access_roles IS DISTINCT FROM %s::text[]"
    ).format(sql.Identifier(table_name(variant)))
    with conn.transaction(), conn.cursor() as cur:
        cur.executemany(
            query, [(list(r.after), r.chunk_id, doc_id, list(r.after)) for r in changed]
        )
    return len(changed)


def distribution(role_sets: Sequence[Sequence[str]]) -> Counter[str]:
    """Rows per role set, keyed like "analyst,risk"."""
    return Counter(",".join(roles) for roles in role_sets)
