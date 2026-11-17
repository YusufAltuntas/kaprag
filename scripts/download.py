"""Download the annual-report PDFs listed in data/manifest.csv into data/raw/.

Run:
  uv run python scripts/download.py                    # download missing files (idempotent)
  uv run python scripts/download.py --update-manifest  # also record sha256 and page counts
  uv run python scripts/download.py --verify           # offline: check files against sha256
"""

import argparse
import sys

from kaprag.corpus import (
    DEFAULT_MANIFEST,
    DEFAULT_RAW_DIR,
    check_document,
    read_manifest,
    write_manifest,
)
from kaprag.download import Status, fetch_all, make_client, with_file_facts


def verify() -> int:
    docs = read_manifest(DEFAULT_MANIFEST)
    failures = [(d.doc_id, err) for d in docs if (err := check_document(d, DEFAULT_RAW_DIR))]
    for doc_id, err in failures:
        print(f"FAIL {doc_id}: {err}")
    if failures:
        print(f"verify: {len(failures)}/{len(docs)} documents failed")
        return 1
    print(f"verify: all {len(docs)} documents ok")
    return 0


def download(update_manifest: bool) -> int:
    docs = read_manifest(DEFAULT_MANIFEST)
    with make_client() as client:
        outcomes = fetch_all(docs, DEFAULT_RAW_DIR, client, check_sha256=not update_manifest)
    for o in outcomes:
        print(f"{o.status.value:<10} {o.doc_id}: {o.detail}")
    if update_manifest:
        write_manifest(with_file_facts(docs, DEFAULT_RAW_DIR), DEFAULT_MANIFEST)
        print(f"manifest updated: {DEFAULT_MANIFEST}")
    failed = [o for o in outcomes if o.status is Status.FAILED]
    downloaded = sum(o.status is Status.DOWNLOADED for o in outcomes)
    skipped = sum(o.status is Status.SKIPPED for o in outcomes)
    print(f"downloaded {downloaded}, skipped {skipped}, failed {len(failed)}")
    for o in failed:
        print(f"FAILED {o.doc_id}: {o.detail}")
    return 1 if failed else 0


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--verify", action="store_true", help="offline sha256 check, no network")
    mode.add_argument(
        "--update-manifest",
        action="store_true",
        help="after downloading, write sha256 and page counts of local files to the manifest",
    )
    args = parser.parse_args()
    return verify() if args.verify else download(args.update_manifest)


if __name__ == "__main__":
    sys.exit(main())
