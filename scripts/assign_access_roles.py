"""Assign `access_roles` to the rows already in the chunk tables (simulated access model).

Run:
  uv run python scripts/assign_access_roles.py --dry-run                     # both variants
  uv run python scripts/assign_access_roles.py --variant fixed
  uv run python scripts/assign_access_roles.py --variant structure --schema acl_scratch

Recomputes each row's roles from its stored `section_path` / `chunk_type` / page span
and the document's Docling rows (`kaprag.access.rules`), the same rule a new index
build applies, and updates only `access_roles`, only where it changes, in one
transaction per document; rerunning is a no-op. No re-chunking, no embedding, no GPU.
`--dry-run` only reads and prints. Prints the before/after distribution per document
and in total. `--schema` works on that schema's copy of the tables instead of the
search_path's (both tables must exist there). A document without complete Docling
output fails (its rows are left unchanged) and the exit code is 1.
"""

import argparse
import sys
from collections import Counter

from dotenv import load_dotenv

from kaprag.access.assign import apply_roles, distribution, document_roles
from kaprag.corpus import DEFAULT_MANIFEST, REPO_ROOT, read_manifest
from kaprag.index.build import MissingInput, load_document_access
from kaprag.index.store import VARIANTS, connect, row_counts, table_name

ROLE_SETS = ("public", "analyst,risk", "risk")


def describe(counts: Counter[str]) -> str:
    total = sum(counts.values()) or 1
    parts = [f"{key or '{}'} {counts[key]} ({counts[key] / total:.1%})" for key in ROLE_SETS]
    parts += [f"{key or '{}'} {n} (?)" for key, n in sorted(counts.items()) if key not in ROLE_SETS]
    return ", ".join(parts)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv(REPO_ROOT / ".env", override=False)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--variant", choices=VARIANTS, action="append", help="default: both")
    parser.add_argument("--schema", default=None, help="schema of the chunk tables")
    parser.add_argument("--doc", action="append", help="doc_id (repeatable); default all")
    parser.add_argument("--dry-run", action="store_true", help="compute and print, write nothing")
    args = parser.parse_args()

    docs = read_manifest(DEFAULT_MANIFEST)
    if args.doc:
        if unknown := sorted(set(args.doc) - {d.doc_id for d in docs}):
            parser.error(f"unknown doc_id(s): {', '.join(unknown)}")
        docs = [d for d in docs if d.doc_id in args.doc]
    mode = "DRY RUN (nothing written)" if args.dry_run else "WRITE"
    failures = 0
    with connect(schema=args.schema) as conn:
        where = conn.execute("SELECT current_schemas(false)").fetchone()[0]
        print(f"{mode}; schema search path {where}")
        for variant in args.variant or VARIANTS:
            table = table_name(variant)
            indexed = row_counts(conn, variant)
            before_total, after_total, updated_total = Counter(), Counter(), 0
            print(f"\n== {table}")
            for doc in docs:
                if doc.doc_id not in indexed:
                    print(f"{doc.doc_id}: no rows, skipped")
                    continue
                try:
                    access = load_document_access(doc)
                except MissingInput as e:
                    print(f"{doc.doc_id}: FAILED: {e}")
                    failures += 1
                    continue
                rows = document_roles(conn, variant, doc.doc_id, access)
                before = distribution([r.before for r in rows])
                after = distribution([r.after for r in rows])
                changed = sum(r.before != r.after for r in rows)
                updated = 0 if args.dry_run else apply_roles(conn, variant, doc.doc_id, rows)
                before_total += before
                after_total += after
                updated_total += updated
                print(f"{doc.doc_id}: {len(rows)} rows; {changed} to change, {updated} updated")
                print(f"    before: {describe(before)}")
                print(f"    after:  {describe(after)}")
            conn.rollback()  # end the read transaction of the last document
            print(f"TOTAL {table}: {sum(after_total.values())} rows, {updated_total} updated")
            print(f"    before: {describe(before_total)}")
            print(f"    after:  {describe(after_total)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
