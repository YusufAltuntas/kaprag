"""Command line for `scripts/export_langfuse.py`: load a finished run into Langfuse.

    uv run python scripts/export_langfuse.py --run runs/<folder> --dry-run
    uv run python scripts/export_langfuse.py --run runs/<folder> [--limit N]

One-way and display-only: the run's JSONL stays the source of truth. A trace that
already exists in Langfuse (same deterministic id) is skipped, since Langfuse treats
ingested observations as immutable and would store a re-sent one twice; scores are
always re-sent with the same id, name and timestamp, which Langfuse overwrites.
The existence check only sees ingested data, so re-exporting within Langfuse's
ingestion lag (about 15-30 s), or after a timed-out send the server did ingest,
duplicates those traces.

Exit codes: 0 everything exported (or built, with --dry-run); 1 some lines or traces
could not be exported (each reported on stderr, the rest is exported); 2 usage error,
not a run folder, missing credentials or the SDK is not installed.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol

from kaprag.observability.langfuse_mapping import (
    RunExport,
    RunFolderError,
    ScoreItem,
    TraceItem,
    load_run,
)


class Sink(Protocol):
    def trace_exists(self, trace_id: str) -> bool: ...
    def send_trace(self, trace: TraceItem) -> None: ...
    def send_scores(self, scores: list[ScoreItem]) -> None: ...
    def flush(self) -> None: ...
    def close(self) -> None: ...


def _default_sink_factory() -> Sink:
    from kaprag.observability.langfuse_sink import LangfuseSettings, LangfuseSink

    return LangfuseSink(LangfuseSettings.from_env())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a finished kaprag run to Langfuse.",
        epilog="Traces already in Langfuse are skipped and scores overwritten; re-exporting "
        "within Langfuse's ingestion lag (about 15-30 s) duplicates traces.",
    )
    parser.add_argument("--run", required=True, type=Path, help="run folder: runs/<folder>")
    parser.add_argument("--limit", type=int, default=None, help="only the first N questions")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="build everything and print counts; no network, no SDK, no credentials",
    )
    return parser


def export_run(export: RunExport, sink: Sink) -> tuple[dict[str, int], list[str]]:
    """Send every trace (unless it already exists) and its scores; never raises per trace."""
    stats = {"sent": 0, "skipped_existing": 0, "failed": 0, "scores": 0}
    errors: list[str] = []
    scores_by_trace: dict[str, list[ScoreItem]] = {}
    for score in export.scores:
        scores_by_trace.setdefault(score.trace_id, []).append(score)
    for trace in export.traces:
        try:
            if sink.trace_exists(trace.id):
                stats["skipped_existing"] += 1
            else:
                sink.send_trace(trace)
                stats["sent"] += 1
        except Exception as exc:  # one failed trace is reported; the rest goes on
            stats["failed"] += 1
            errors.append(f"{trace.qid} (trace {trace.id}): {type(exc).__name__}: {exc}")
            continue
        scores = scores_by_trace.get(trace.id, [])
        try:
            sink.send_scores(scores)
            stats["scores"] += len(scores)
        except Exception as exc:
            errors.append(f"{trace.qid} scores: {type(exc).__name__}: {exc}")
    return stats, errors


def main(
    argv: Sequence[str] | None = None,
    *,
    sink_factory: Callable[[], Any] = _default_sink_factory,
) -> int:
    args = build_parser().parse_args(argv)
    if args.limit is not None and args.limit < 1:
        print("error: --limit must be at least 1", file=sys.stderr)
        return 2
    try:
        export = load_run(args.run, limit=args.limit)
    except RunFolderError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    for message in export.errors:
        print(f"error: {message}", file=sys.stderr)

    if args.dry_run:
        counts = " ".join(f"{k}={v}" for k, v in export.counts().items())
        print(f"dry run (nothing sent): run={export.run_key} {counts}")
        return 1 if export.errors else 0

    try:
        sink = sink_factory()
    except Exception as exc:  # missing credentials / SDK / unreachable server
        print(f"error: {exc}", file=sys.stderr)
        return 2
    try:
        stats, send_errors = export_run(export, sink)
    finally:
        try:
            sink.flush()
        finally:
            sink.close()
    for message in send_errors:
        print(f"error: {message}", file=sys.stderr)
    counts = export.counts()
    print(
        f"exported: run={export.run_key} traces={counts['traces']} "
        f"observations={counts['observations']} generations={counts['generations']} "
        f"sent={stats['sent']} skipped_existing={stats['skipped_existing']} "
        f"failed={stats['failed']} scores={stats['scores']} "
        f"errors={len(export.errors) + len(send_errors)}"
    )
    return 1 if export.errors or send_errors else 0
