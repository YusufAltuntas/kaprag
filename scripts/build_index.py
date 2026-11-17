"""Build a chunk index variant: chunk -> embed (bge-m3, GPU) -> write to Postgres.

Run:
  uv run python scripts/build_index.py --variant fixed                       # all documents
  uv run python scripts/build_index.py --variant structure --doc garanti-2023 --doc akbank-2023
  uv run python scripts/build_index.py --variant structure --skip-existing   # resume

`fixed` reads data/parsed/pymupdf, `structure` reads data/parsed/docling (only documents
with a completion marker). Each document's rows are replaced in one transaction, so a
document is either fully indexed or untouched; rerunning is idempotent. After an
interruption, `--skip-existing` builds only documents with no rows in the table (use
`--doc` to force a rebuild of specific ones). A failing or missing document is logged
and the run continues; the exit code is 1 if any document failed. Needs the `ml` group
and the docker compose Postgres.
"""

import argparse
import logging
import sys
import time

from dotenv import load_dotenv

from kaprag.corpus import DEFAULT_MANIFEST, REPO_ROOT, read_manifest
from kaprag.index.build import MissingInput, chunk_document, chunk_stats
from kaprag.index.embed import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MAX_SEQ_LENGTH,
    Embedder,
    embed_model_name,
)
from kaprag.index.store import (
    VARIANTS,
    connect,
    ensure_schema,
    replace_document,
    row_counts,
    table_name,
)

log = logging.getLogger("build_index")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv(REPO_ROOT / ".env", override=False)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    parser.add_argument("--doc", action="append", help="doc_id to index (repeatable); default all")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="skip documents that already have rows in the variant's table",
    )
    parser.add_argument("--model", default=None, help="embedding model (default: EMBED_MODEL)")
    parser.add_argument("--device", default="cuda", help="cuda, cuda:N or cpu")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-seq-length", type=int, default=DEFAULT_MAX_SEQ_LENGTH)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
    for noisy in ("sentence_transformers", "transformers", "httpx", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    docs = read_manifest(DEFAULT_MANIFEST)
    if args.doc:
        known = {d.doc_id for d in docs}
        if unknown := sorted(set(args.doc) - known):
            parser.error(f"unknown doc_id(s): {', '.join(unknown)}")
        docs = [d for d in docs if d.doc_id in args.doc]
    model_name = args.model or embed_model_name()

    with connect() as conn:
        ensure_schema(conn)
        if args.skip_existing:
            existing = row_counts(conn, args.variant)
            skipped = [d.doc_id for d in docs if d.doc_id in existing]
            if skipped:
                log.info("skipping %d already indexed: %s", len(skipped), ", ".join(skipped))
            docs = [d for d in docs if d.doc_id not in existing]

        failures: dict[str, str] = {}
        if docs:
            from kaprag.chunking.base import load_token_spans

            start = time.monotonic()
            token_spans = load_token_spans(model_name)
            embedder = Embedder(
                model_name,
                device=args.device,
                batch_size=args.batch_size,
                max_seq_length=args.max_seq_length,
            )
            log.info(
                "%s loaded on %s (batch %d, max_seq_length %d) in %.1f s",
                model_name,
                args.device,
                args.batch_size,
                args.max_seq_length,
                time.monotonic() - start,
            )
            embedded, embed_seconds = 0, 0.0
            for i, doc in enumerate(docs, start=1):
                prefix = f"[{i}/{len(docs)}] {doc.doc_id}"
                try:
                    t0 = time.monotonic()
                    chunks = chunk_document(args.variant, doc, token_spans)
                    t1 = time.monotonic()
                    log.info(
                        "%s: chunked in %.1f s: %s", prefix, t1 - t0, chunk_stats(chunks).describe()
                    )
                    vectors = embedder.embed_documents([c.text for c in chunks])
                    t2 = time.monotonic()
                    written = replace_document(conn, args.variant, doc, chunks, vectors)
                    t3 = time.monotonic()
                except MissingInput as e:
                    failures[doc.doc_id] = f"skipped, missing input: {e}"
                    log.error("%s: SKIPPED: %s", prefix, e)
                    continue
                except Exception as e:  # keep going with the other documents
                    conn.rollback()
                    failures[doc.doc_id] = f"{type(e).__name__}: {e}"
                    log.error("%s: FAILED: %s: %s", prefix, type(e).__name__, e)
                    continue
                embedded += len(chunks)
                embed_seconds += t2 - t1
                log.info(
                    "%s: %d rows written; embedding %.1f s (%.1f chunks/s), write %.1f s",
                    prefix,
                    written,
                    t2 - t1,
                    len(chunks) / (t2 - t1) if t2 > t1 else float("nan"),
                    t3 - t2,
                )
            peak = embedder.peak_memory_gb()
            log.info(
                "embedded %d chunks in %.1f s (%.1f chunks/s), peak CUDA memory %s, total %.1f min",
                embedded,
                embed_seconds,
                embedded / embed_seconds if embed_seconds else float("nan"),
                f"{peak:.2f} GB" if peak is not None else "n/a",
                (time.monotonic() - start) / 60,
            )

        counts = row_counts(conn, args.variant)
    table = table_name(args.variant)
    for doc_id, n in counts.items():
        log.info("%s rows %s: %d", table, doc_id, n)
    log.info("%s rows total: %d in %d documents", table, sum(counts.values()), len(counts))
    for doc_id, error in failures.items():
        log.error("FAILED %s: %s", doc_id, error)
    if failures:
        log.error("%d document(s) failed", len(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
