"""Resumable, failure-isolating driver for slow page-range parsers (Docling).

Files under `out_dir` for a document `<doc_id>` converted over pages `first..last`:

- `<doc_id>.parts/<first>-<last>.jsonl` - part rows of one page range, written
  atomically when that range finishes. A resumed run skips ranges whose part exists.
- `<doc_id>.jsonl` - the final rows (with `section_path`), assembled from the parts
  once every range of the document succeeded, written atomically.
- `<doc_id>.done.json` - the completion marker, written after the final file:
  `{"doc_id", "first_page", "last_page", "rows"}`. A document counts as complete
  only if the marker exists and covers the requested pages, so a page subset (e.g.
  a sample run) is never mistaken for a whole document. The parts directory is
  removed after the marker is written.

A failing range is logged and recorded; the run moves on to the next range and
document, and the document stays incomplete until a later run fills the gap.
"""

import json
import logging
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from kaprag.parsing.docling_parser import assign_section_paths
from kaprag.parsing.outputs import iter_jsonl, write_jsonl_atomic

log = logging.getLogger(__name__)

# (pdf_path, first_page, last_page) -> part rows; raises on failure.
ConvertFn = Callable[[Path, int, int], list[dict]]


@dataclass(frozen=True)
class Job:
    """One document to convert, over PDF pages `first..last` (1-based, inclusive)."""

    doc_id: str
    pdf_path: Path
    first: int
    last: int


@dataclass(frozen=True)
class Failure:
    doc_id: str
    first: int | None
    last: int | None
    error: str


@dataclass
class RunSummary:
    completed: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failures: list[Failure] = field(default_factory=list)
    pages_converted: int = 0
    convert_seconds: float = 0.0


def page_ranges(first: int, last: int, size: int) -> list[tuple[int, int]]:
    """Split `first..last` into consecutive inclusive ranges of at most `size` pages."""
    if size < 1:
        raise ValueError("size must be >= 1")
    return [(a, min(a + size - 1, last)) for a in range(first, last + 1, size)]


def final_path(out_dir: Path, doc_id: str) -> Path:
    return out_dir / f"{doc_id}.jsonl"


def marker_path(out_dir: Path, doc_id: str) -> Path:
    return out_dir / f"{doc_id}.done.json"


def part_path(out_dir: Path, doc_id: str, first: int, last: int) -> Path:
    return out_dir / f"{doc_id}.parts" / f"{first:04d}-{last:04d}.jsonl"


def is_complete(out_dir: Path, job: Job) -> bool:
    """True if `job`'s final file exists and its marker covers exactly the job's pages."""
    marker = marker_path(out_dir, job.doc_id)
    if not (marker.is_file() and final_path(out_dir, job.doc_id).is_file()):
        return False
    try:
        meta = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (meta.get("first_page"), meta.get("last_page")) == (job.first, job.last)


def run_jobs(
    jobs: list[Job],
    out_dir: Path,
    convert: ConvertFn,
    pages_per_batch: int,
    on_failure: Callable[[], None] | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> RunSummary:
    """Convert every job's pages range by range, resuming from what is on disk."""
    summary = RunSummary()
    todo: list[tuple[Job, list[tuple[int, int]], list[tuple[int, int]]]] = []
    for job in jobs:
        if is_complete(out_dir, job):
            summary.skipped.append(job.doc_id)
            log.info("%s: already complete, skipped", job.doc_id)
            continue
        ranges = page_ranges(job.first, job.last, pages_per_batch)
        pending = [r for r in ranges if not part_path(out_dir, job.doc_id, *r).is_file()]
        todo.append((job, ranges, pending))
    total_pages = sum(b - a + 1 for _, _, pending in todo for a, b in pending)
    log.info(
        "%d documents to do (%d already complete), %d pages to convert",
        len(todo),
        len(summary.skipped),
        total_pages,
    )

    failed_pages = 0
    for job, ranges, pending in todo:
        if not job.pdf_path.is_file():
            summary.failures.append(Failure(job.doc_id, None, None, f"PDF missing: {job.pdf_path}"))
            log.error("%s: PDF missing: %s", job.doc_id, job.pdf_path)
            continue
        doc_failed = False
        for first, last in pending:
            start = clock()
            try:
                rows = convert(job.pdf_path, first, last)
            except Exception as e:  # isolate the failure to this range
                doc_failed = True
                failed_pages += last - first + 1
                error = f"{type(e).__name__}: {e}"
                summary.failures.append(Failure(job.doc_id, first, last, error))
                log.error("%s pages %d-%d FAILED: %s", job.doc_id, first, last, error)
                if on_failure is not None:
                    on_failure()
                continue
            try:
                write_jsonl_atomic(part_path(out_dir, job.doc_id, first, last), rows)
            except Exception as e:  # isolate the failure to this range
                doc_failed = True
                failed_pages += last - first + 1
                error = f"writing part failed: {type(e).__name__}: {e}"
                summary.failures.append(Failure(job.doc_id, first, last, error))
                log.error("%s pages %d-%d FAILED: %s", job.doc_id, first, last, error)
                continue
            seconds = clock() - start
            pages = last - first + 1
            summary.pages_converted += pages
            summary.convert_seconds += seconds
            rate = summary.pages_converted / summary.convert_seconds if seconds > 0 else 0.0
            remaining = total_pages - summary.pages_converted - failed_pages
            eta = remaining / rate if rate > 0 else float("nan")
            log.info(
                "%s pages %d-%d: %d rows, %.1f s, %.2f pages/s | run %d/%d pages, "
                "%.2f pages/s, eta %.1f min",
                job.doc_id,
                first,
                last,
                len(rows),
                seconds,
                pages / seconds if seconds else 0.0,
                summary.pages_converted,
                total_pages,
                rate,
                eta / 60,
            )
        if doc_failed:
            continue
        try:
            rows = _assemble(out_dir, job, ranges)
        except Exception as e:  # isolate the failure to this document
            error = f"assembly failed: {type(e).__name__}: {e}"
            summary.failures.append(Failure(job.doc_id, None, None, error))
            log.error("%s FAILED: %s", job.doc_id, error)
            continue
        summary.completed.append(job.doc_id)
        log.info("%s: complete, %d rows -> %s", job.doc_id, rows, final_path(out_dir, job.doc_id))
    return summary


def _assemble(out_dir: Path, job: Job, ranges: list[tuple[int, int]]) -> int:
    """Join the parts of a fully converted job into the final file and mark it complete.

    Parts left over from a run with another range size are not used and are removed.
    """
    part_rows = (row for r in ranges for row in iter_jsonl(part_path(out_dir, job.doc_id, *r)))
    count = write_jsonl_atomic(final_path(out_dir, job.doc_id), assign_section_paths(part_rows))
    marker = {"doc_id": job.doc_id, "first_page": job.first, "last_page": job.last, "rows": count}
    tmp = marker_path(out_dir, job.doc_id).with_suffix(".tmp")
    tmp.write_text(json.dumps(marker) + "\n", encoding="utf-8")
    tmp.replace(marker_path(out_dir, job.doc_id))
    # The marker is written, so the parts are no longer needed; a leftover is harmless.
    parts_dir = out_dir / f"{job.doc_id}.parts"
    try:
        shutil.rmtree(parts_dir)
    except OSError as e:
        log.warning("%s: could not remove %s: %s", job.doc_id, parts_dir, e)
    return count
