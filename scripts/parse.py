"""Parse the manifest's PDFs with PyMuPDF or Docling into JSON Lines.

Run:
  uv run python scripts/parse.py --parser pymupdf                # all documents, idempotent
  uv run python scripts/parse.py --parser docling                # all documents, resumable
  uv run python scripts/parse.py --parser docling --doc akbank-2024 --pages 150-169 \
      --out-dir runs/2026-11-17-docling-sample/docling          # a sample

Outputs go to data/parsed/<parser>/<doc_id>.jsonl unless --out-dir is given. A Docling
run can be interrupted and restarted with the same arguments: finished documents and
page ranges are skipped. A failing page range is logged and the run continues; the exit
code is 1 if anything failed.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

from kaprag.corpus import DEFAULT_MANIFEST, DEFAULT_RAW_DIR, raw_path, read_manifest
from kaprag.parsing.batch_runner import Job, is_complete, run_jobs
from kaprag.parsing.outputs import DEFAULT_PARSED_DIR

log = logging.getLogger("parse")


def parse_page_range(value: str) -> tuple[int, int]:
    try:
        first, last = (int(x) for x in value.split("-", 1))
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected FIRST-LAST, got {value!r}") from None
    if not 1 <= first <= last:
        raise argparse.ArgumentTypeError(f"invalid page range {value!r}")
    return first, last


def run_pymupdf(docs, out_dir: Path) -> int:
    from kaprag.parsing.pymupdf_parser import parse_document

    failed = 0
    for doc in docs:
        out = out_dir / f"{doc.doc_id}.jsonl"
        try:
            pages = parse_document(raw_path(doc.doc_id, DEFAULT_RAW_DIR), out)
        except Exception as e:  # keep going with the other documents
            failed += 1
            log.error("%s FAILED: %s: %s", doc.doc_id, type(e).__name__, e)
            continue
        log.info("%s: %d pages -> %s", doc.doc_id, pages, out)
    log.info("pymupdf: %d documents ok, %d failed", len(docs) - failed, failed)
    return 1 if failed else 0


def run_docling(docs, out_dir: Path, args) -> int:
    from kaprag.parsing.docling_parser import DoclingConverter

    jobs = []
    for doc in docs:
        if doc.pages is None:
            log.error("%s: page count missing from the manifest", doc.doc_id)
            return 1
        first, last = args.pages or (1, doc.pages)
        if last > doc.pages:
            log.error("%s: page range ends after page %d", doc.doc_id, doc.pages)
            return 1
        jobs.append(Job(doc.doc_id, raw_path(doc.doc_id, DEFAULT_RAW_DIR), first, last))
    if all(is_complete(out_dir, job) for job in jobs):
        log.info("docling: all %d documents already complete in %s", len(jobs), out_dir)
        return 0

    start = time.monotonic()
    converter = DoclingConverter(device=args.device, model_batch_size=args.model_batch_size)
    log.info("docling: models loaded on %s in %.1f s", args.device, time.monotonic() - start)
    summary = run_jobs(
        jobs, out_dir, converter, args.pages_per_batch, on_failure=converter.release_memory
    )
    peak = converter.peak_memory_gb()
    rate = summary.pages_converted / summary.convert_seconds if summary.convert_seconds else 0
    log.info(
        "docling: %d documents completed, %d already complete, %d pages converted in %.1f s "
        "(%.2f s/page), peak CUDA memory %s, total %.1f min",
        len(summary.completed),
        len(summary.skipped),
        summary.pages_converted,
        summary.convert_seconds,
        1 / rate if rate else float("nan"),
        f"{peak:.2f} GB" if peak is not None else "n/a",
        (time.monotonic() - start) / 60,
    )
    for f in summary.failures:
        pages = f"pages {f.first}-{f.last}" if f.first is not None else "document"
        log.error("FAILED %s %s: %s", f.doc_id, pages, f.error)
    return 1 if summary.failures else 0


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--parser", required=True, choices=["pymupdf", "docling"])
    parser.add_argument("--doc", action="append", help="doc_id to parse (repeatable); default all")
    parser.add_argument("--out-dir", type=Path, help="default: data/parsed/<parser>")
    parser.add_argument(
        "--pages",
        type=parse_page_range,
        help="docling only: PDF page range FIRST-LAST (1-based); needs one --doc and --out-dir",
    )
    parser.add_argument("--device", default="cuda", help="docling: cuda, cuda:N, cpu or auto")
    parser.add_argument(
        "--pages-per-batch",
        type=int,
        default=20,
        help="docling: pages converted per call; the unit of resume and failure isolation",
    )
    parser.add_argument(
        "--model-batch-size",
        type=int,
        default=4,
        help="docling: layout/table model batch size (GPU memory)",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
    for noisy in ("docling", "docling_core", "docling_ibm_models", "transformers", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if args.pages and (args.parser != "docling" or not args.doc or len(args.doc) != 1):
        parser.error("--pages needs --parser docling and exactly one --doc")
    if args.pages and args.out_dir is None:
        parser.error("--pages needs --out-dir, so a partial document never lands in data/parsed")

    docs = read_manifest(DEFAULT_MANIFEST)
    if args.doc:
        known = {d.doc_id for d in docs}
        if unknown := sorted(set(args.doc) - known):
            parser.error(f"unknown doc_id(s): {', '.join(unknown)}")
        docs = [d for d in docs if d.doc_id in args.doc]
    out_dir = args.out_dir or DEFAULT_PARSED_DIR / args.parser
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.parser == "pymupdf":
        return run_pymupdf(docs, out_dir)
    return run_docling(docs, out_dir, args)


if __name__ == "__main__":
    sys.exit(main())
