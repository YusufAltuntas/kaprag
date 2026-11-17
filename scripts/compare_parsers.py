"""Compare PyMuPDF and Docling output for one table page per bank (or chosen pages).

Run:
  uv run python scripts/compare_parsers.py                          # one table page per bank
  uv run python scripts/compare_parsers.py --doc akbank-2024 --page 153
  uv run python scripts/compare_parsers.py --docling-dir runs/<run>/docling \
      --out runs/<run>/compare.txt

Reads the parsed outputs on disk only. By default it picks, for each bank, the first
manifest document with Docling output containing a table, and its page with the most
tables. `--out` must be outside the repository or under runs/ (git-ignored).
"""

import argparse
import sys
from pathlib import Path

from kaprag.corpus import DEFAULT_MANIFEST, REPO_ROOT, read_manifest
from kaprag.parsing.compare import compare_page, default_choices
from kaprag.parsing.outputs import DEFAULT_PARSED_DIR


def check_out_path(path: Path) -> Path:
    resolved = path.resolve()
    if resolved.is_relative_to(REPO_ROOT) and not resolved.is_relative_to(REPO_ROOT / "runs"):
        raise argparse.ArgumentTypeError(f"--out must be under runs/ or outside the repo: {path}")
    return resolved


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pymupdf-dir", type=Path, default=DEFAULT_PARSED_DIR / "pymupdf")
    parser.add_argument("--docling-dir", type=Path, default=DEFAULT_PARSED_DIR / "docling")
    parser.add_argument("--doc", action="append", help="doc_id (repeatable, pairs with --page)")
    parser.add_argument("--page", type=int, action="append", help="1-based PDF page per --doc")
    parser.add_argument("--out", type=Path, help="also write the report to this file")
    args = parser.parse_args()

    if (args.doc is None) != (args.page is None) or (
        args.doc is not None and len(args.doc) != len(args.page)
    ):
        parser.error("give --doc and --page together, once per page")
    out = None
    if args.out is not None:
        try:
            out = check_out_path(args.out)
        except argparse.ArgumentTypeError as e:
            parser.error(str(e))

    if args.doc:
        choices = list(zip(args.doc, args.page, strict=True))
    else:
        choices = default_choices(read_manifest(DEFAULT_MANIFEST), args.docling_dir)
        if not choices:
            print(f"no Docling output with tables in {args.docling_dir}", file=sys.stderr)
            return 1
    report = "\n\n".join(
        compare_page(doc_id, page, args.pymupdf_dir, args.docling_dir) for doc_id, page in choices
    )
    print(report)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report + "\n", encoding="utf-8")
        print(f"\nwritten to {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
