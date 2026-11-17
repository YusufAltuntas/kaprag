"""Apply an agent audit's decisions to the gold file.

    uv run python scripts/apply_audit.py --decisions runs/2026-11-18-gold-audit/decisions.json
    uv run python scripts/apply_audit.py --decisions <file> --questions <copy.jsonl> --dry-run
    uv run python scripts/apply_audit.py --decisions <file> --sync-translations <questions_en.jsonl>

Sets the `audit` field (never `verified`), applies fixes, removes unsupported extra
evidence pages, moves rejected rows to rejected.jsonl next to the questions file and
recomputes union-built derived evidence. With --sync-translations, translated rows
then take their source's final evidence and expected value/unit/tolerance (refused if
a source is rejected or missing). See `kaprag.eval.audit` for the rules.

Exit codes: 0 applied (or dry run), 1 the decisions cannot be applied (nothing
written), 2 refused (verified / human rows, already audited rows without --force,
a translation whose source is rejected or missing, bad arguments; nothing written).
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

from kaprag.eval.audit import AuditError, AuditRefused, apply_audit_file, format_summary
from kaprag.eval.schema import DEFAULT_QUESTIONS, GoldSetError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Apply gold-set audit decisions.")
    parser.add_argument("--decisions", type=Path, required=True, help="decisions.json")
    parser.add_argument(
        "--questions",
        type=Path,
        default=DEFAULT_QUESTIONS,
        help="gold JSONL file (default: eval/gold/questions.jsonl)",
    )
    parser.add_argument("--date", default=None, help="audit date YYYY-MM-DD (default: today)")
    parser.add_argument("--dry-run", action="store_true", help="print the summary, write nothing")
    parser.add_argument(
        "--force", action="store_true", help="re-apply to rows that already carry an audit"
    )
    parser.add_argument(
        "--sync-translations",
        type=Path,
        default=None,
        metavar="PATH",
        help="translated gold file (e.g. eval/gold/questions_en.jsonl) to sync to the sources",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    day = args.date or date.today().isoformat()
    try:
        date.fromisoformat(day)
        if len(day) != 10:
            raise ValueError
    except ValueError:
        print(f"--date must be YYYY-MM-DD, got {day!r}", file=sys.stderr)
        return 2
    try:
        outcome = apply_audit_file(
            args.questions,
            args.decisions,
            date=day,
            force=args.force,
            dry_run=args.dry_run,
            translations_path=args.sync_translations,
        )
    except AuditRefused as exc:
        print(f"refused, nothing written: {exc}", file=sys.stderr)
        return 2
    except (AuditError, GoldSetError) as exc:
        print(f"error, nothing written: {exc}", file=sys.stderr)
        return 1
    print(format_summary(outcome))
    if not args.dry_run:
        print(f"written: {args.questions}")
        if outcome.translations_synced:
            print(f"written: {args.sync_translations}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
