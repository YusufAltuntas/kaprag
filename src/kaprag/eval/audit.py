"""Apply an agent audit's decisions to the gold file (`scripts/apply_audit.py`).

The decisions file (`runs/<date>-gold-audit/decisions.json`) maps question id to
`{action, set, remove_evidence, reason}`; its `rules` block explains each action.
Per question:

- `pass`: drop the listed `remove_evidence` pages (never the first evidence entry);
  `audit.status = passed`, note empty unless pages were removed.
- `fix`: apply each `set` key (`evidence`, `expected.answer`, `expected.value`,
  `expected.tolerance`; a dotted key addresses the nested field), then
  `remove_evidence`; `audit.status = fixed`, note = the shortened reason.
- `flag`: only `audit.status = flagged` (note = reason); the row is otherwise unchanged
  and stays outside the audited subset.
- `reject`: the row moves to `rejected.jsonl` next to the questions file (the review
  app's file), the reason appended to `notes`.

Then every remaining derived question (`derived_from` non-empty, not flagged) whose
current evidence equals, as a SET of `(doc, page)`, the union of its sources' ORIGINAL
evidence is recomputed from their FINAL evidence, when every source is still present:
the row keeps its own order of the pages still in the final union (so its first entry
stays first where possible) and the final union's new pages are appended. The set
rule matters because `kaprag.gold.evidence.propagate` puts the derived draft's own
page first, so a union-built row is rarely in the sources' order.

Optionally, translated rows (`eval/gold/questions_en.jsonl`, each with
`derived_from: [<source id>]`) are synced afterwards: a translation whose `evidence`
or `expected.value` / `unit` / `tolerance` differ from its source's FINAL row takes
the source's values; its own text (`question`, `expected.answer`, `key_points`,
`notes`) and everything else stay. A translation whose source is rejected or missing
refuses the whole run.

An audit is an agent's statement, not human verification: nothing here sets
`verified`/`verified_at`, and rows that are verified or written by a human are never
changed (the whole run is refused). Rows already carrying an `audit` are refused too
unless `force` (re-running with the same decisions then gives the same files).
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from kaprag.eval.review import load_for_review, rejected_path_for, with_reject_reason
from kaprag.eval.schema import QUESTION_TYPES, Question, save_questions

ACTIONS = ("pass", "fix", "flag", "reject")
SETTABLE = ("evidence", "expected.answer", "expected.value", "expected.tolerance")
NOTE_MAX_CHARS = 200
AUDIT_STATUS = {"pass": "passed", "fix": "fixed", "flag": "flagged"}

EvidenceKey = tuple[str, int | None]


class AuditError(ValueError):
    """The decisions cannot be applied (malformed decision, unknown id, invalid result)."""


class AuditRefused(AuditError):
    """Applying would touch a verified / human-written row, or re-audit an audited one."""


@dataclass(frozen=True)
class Decision:
    id: str
    action: str
    set: dict[str, Any]
    remove_evidence: list[EvidenceKey]
    reason: str


@dataclass
class AuditOutcome:
    questions: list[Question]  # remaining rows, file order
    rejected: list[Question]  # rows to append to rejected.jsonl
    counts: Counter[str] = field(default_factory=Counter)  # decisions per action
    evidence_removed: int = 0
    derived_recomputed: list[str] = field(default_factory=list)  # ids whose evidence changed
    already_rejected: list[str] = field(default_factory=list)  # reject ids already moved
    translations: list[Question] | None = None  # synced translated rows, file order
    translations_synced: list[str] = field(default_factory=list)  # ids updated
    translation_sources: dict[str, str] = field(default_factory=dict)  # synced id -> source
    dry_run: bool = False


# --- decisions file ------------------------------------------------------------------


def _evidence_key(item: Any, where: str) -> EvidenceKey:
    if not isinstance(item, dict) or set(item) != {"doc", "page"}:
        raise AuditError(f"{where}: evidence entries are {{doc, page}} objects, got {item!r}")
    return item["doc"], item["page"]


def _parse_decision(key: str, raw: Any) -> Decision:
    if not isinstance(raw, dict):
        raise AuditError(f"decision {key!r} is not an object")
    qid = raw.get("id", key)
    if qid != key:
        raise AuditError(f"decision under key {key!r} has id {qid!r}")
    action = raw.get("action")
    if action not in ACTIONS:
        raise AuditError(f"{qid}: unknown action {action!r}; expected one of {ACTIONS}")
    changes = raw.get("set") or {}
    remove = raw.get("remove_evidence") or []
    reason = raw.get("reason") or ""
    if not isinstance(changes, dict) or not isinstance(remove, list):
        raise AuditError(f"{qid}: 'set' must be an object and 'remove_evidence' a list")
    if bad := sorted(set(changes) - set(SETTABLE)):
        raise AuditError(f"{qid}: keys not settable by an audit: {bad}; allowed: {SETTABLE}")
    if action != "fix" and changes:
        raise AuditError(f"{qid}: action {action!r} cannot set fields")
    if action in ("flag", "reject") and remove:
        raise AuditError(f"{qid}: action {action!r} cannot remove evidence")
    if action == "fix" and not changes and not remove:
        raise AuditError(f"{qid}: 'fix' decision with nothing to apply")
    return Decision(
        id=qid,
        action=action,
        set=dict(changes),
        remove_evidence=[_evidence_key(e, qid) for e in remove],
        reason=str(reason),
    )


def parse_decisions(data: Any) -> list[Decision]:
    """Decisions from a parsed decisions file (`{"rules": ..., "decisions": {id: ...}}`)."""
    raw = data.get("decisions") if isinstance(data, dict) else None
    if not isinstance(raw, dict):
        raise AuditError("decisions file needs a 'decisions' object keyed by question id")
    return [_parse_decision(key, value) for key, value in raw.items()]


def load_decisions(path: Path) -> list[Decision]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditError(f"cannot read decisions file {path}: {exc}") from None
    return parse_decisions(data)


# --- applying ------------------------------------------------------------------------


def shorten(text: str, limit: int = NOTE_MAX_CHARS) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _union(lists: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    seen: dict[EvidenceKey, dict[str, Any]] = {}
    for evidence in lists:
        for item in evidence:
            seen.setdefault((item["doc"], item["page"]), item)
    return list(seen.values())


def _keys(evidence: list[dict[str, Any]]) -> set[EvidenceKey]:
    return {(e["doc"], e["page"]) for e in evidence}


def _validate(data: dict[str, Any]) -> Question:
    try:
        return Question.model_validate(data)
    except ValidationError as exc:
        raise AuditError(f"question {data.get('id')!r} would be invalid: {exc}") from None


def _apply_one(q: Question, d: Decision, *, date: str, force: bool) -> tuple[Question, int]:
    """The row after a pass/fix/flag decision and the number of evidence entries removed."""
    data = q.model_dump(mode="json")
    for key, value in d.set.items():
        outer, _, inner = key.partition(".")
        if inner:
            data[outer] = {**data[outer], inner: value}
        else:
            data[outer] = value
    evidence = data["evidence"]
    removed = 0
    for key in d.remove_evidence:
        if evidence and (evidence[0]["doc"], evidence[0]["page"]) == key:
            raise AuditError(f"{q.id}: refusing to remove the first evidence entry {key}")
        kept = [e for e in evidence if (e["doc"], e["page"]) != key]
        if len(kept) == len(evidence) and not force:
            raise AuditError(f"{q.id}: page {key} to remove is not in its evidence")
        removed += len(evidence) - len(kept)
        evidence = kept
    data["evidence"] = evidence
    if d.action == "pass":
        n = len(d.remove_evidence)
        note = f"removed {n} unsupported evidence page(s)" if n else ""
    else:
        note = shorten(d.reason)
    data["audit"] = {"status": AUDIT_STATUS[d.action], "by": "agent", "at": date, "note": note}
    return _validate(data), removed


def _protected(q: Question) -> str | None:
    if q.verified:
        return "verified"
    if q.author == "human":
        return "author human"
    return None


def apply_decisions(
    questions: list[Question],
    decisions: list[Decision],
    *,
    date: str,
    force: bool = False,
    rejected_ids: frozenset[str] | set[str] = frozenset(),
) -> AuditOutcome:
    """Pure: the gold set after the decisions. Raises before returning anything partial.

    `rejected_ids` are ids already in the rejected file: a `reject` decision for an id
    that is there and no longer in `questions` was applied before and is skipped.
    """
    original = {q.id: q for q in questions}
    outcome = AuditOutcome(questions=[], rejected=[])
    targeted: dict[str, Decision] = {}
    for d in decisions:
        if d.id in targeted:
            raise AuditError(f"two decisions for {d.id}")
        if d.id not in original:
            if d.action == "reject" and d.id in rejected_ids:
                outcome.already_rejected.append(d.id)
                outcome.counts[d.action] += 1
                continue
            raise AuditError(f"decision for {d.id}, which is not in the questions file")
        if d.action == "reject" and d.id in rejected_ids:
            raise AuditError(f"{d.id} is already in the rejected file")
        targeted[d.id] = d

    refused = [f"{qid} ({why})" for qid in targeted if (why := _protected(original[qid]))]
    if refused:
        raise AuditRefused(f"refusing to change verified or human rows: {', '.join(refused)}")
    if not force and (done := [qid for qid in targeted if original[qid].audit is not None]):
        raise AuditRefused(
            f"rows already audited (pass --force to re-apply): {', '.join(done[:10])}"
            + (f" and {len(done) - 10} more" if len(done) > 10 else "")
        )

    current: dict[str, Question] = {}
    for q in questions:
        d = targeted.get(q.id)
        if d is None:
            current[q.id] = q
            continue
        outcome.counts[d.action] += 1
        if d.action == "reject":
            outcome.rejected.append(with_reject_reason(q, d.reason))
            continue
        current[q.id], removed = _apply_one(q, d, date=date, force=force)
        outcome.evidence_removed += removed

    for qid, q in current.items():
        sources = q.derived_from
        if not sources or not all(s in current and s in original for s in sources):
            continue
        if q.audit is not None and q.audit.status == "flagged":
            continue  # a flagged row stays unchanged apart from its audit
        evidence = q.model_dump(mode="json")["evidence"]
        was_union = _union([original[s].model_dump(mode="json")["evidence"] for s in sources])
        if _keys(evidence) != _keys(was_union):
            continue
        final_union = _union([current[s].model_dump(mode="json")["evidence"] for s in sources])
        final_keys = _keys(final_union)
        final = _union([[e for e in evidence if (e["doc"], e["page"]) in final_keys], final_union])
        if final == evidence:
            continue
        if why := _protected(q):
            raise AuditRefused(f"refusing to recompute evidence of {qid} ({why})")
        current[qid] = _validate(q.model_dump(mode="json") | {"evidence": final})
        outcome.derived_recomputed.append(qid)

    outcome.questions = list(current.values())
    return outcome


SYNCED_EXPECTED = ("value", "unit", "tolerance")


def sync_translations(
    translations: list[Question],
    final: list[Question],
    *,
    rejected_ids: set[str] | frozenset[str] = frozenset(),
) -> tuple[list[Question], list[str]]:
    """Translated rows with their source's final evidence and expected numbers.

    Returns (rows in input order, ids that changed). Raises `AuditRefused` when a
    translation's source is rejected or missing, or a changed row is verified/human.
    """
    sources = {q.id: q for q in final}
    problems = []
    for t in translations:
        if len(t.derived_from) != 1:
            problems.append(f"{t.id} (derived_from must name one source)")
        elif (sid := t.derived_from[0]) in rejected_ids:
            problems.append(f"{t.id} (source {sid} is rejected)")
        elif sid not in sources:
            problems.append(f"{t.id} (source {sid} is not in the questions file)")
    if problems:
        raise AuditRefused(f"cannot sync translations: {', '.join(problems)}")

    out, synced = [], []
    for t in translations:
        source = sources[t.derived_from[0]]
        data = t.model_dump(mode="json")
        src = source.model_dump(mode="json")
        new = data | {
            "evidence": src["evidence"],
            "expected": data["expected"] | {k: src["expected"][k] for k in SYNCED_EXPECTED},
        }
        if new == data:
            out.append(t)
            continue
        if why := _protected(t):
            raise AuditRefused(f"refusing to sync translation {t.id} ({why})")
        out.append(_validate(new))
        synced.append(t.id)
    return out, synced


def apply_audit_file(
    questions_path: Path,
    decisions_path: Path,
    *,
    date: str,
    force: bool = False,
    dry_run: bool = False,
    translations_path: Path | None = None,
) -> AuditOutcome:
    """Apply a decisions file to a gold file; writes nothing on any error or with `dry_run`.

    With `translations_path`, translated rows are synced to their sources afterwards
    (see `sync_translations`); everything is validated before any file is written.
    The rejected file is written first (as the review app does): a crash in between
    leaves a duplicate row, never a lost one; the translations file is written last.
    """
    questions_path = Path(questions_path)
    questions = load_for_review(questions_path)
    if not questions:
        raise AuditError(f"no questions in {questions_path}")
    rejected_path = rejected_path_for(questions_path)
    rejected = load_for_review(rejected_path)
    outcome = apply_decisions(
        questions,
        load_decisions(decisions_path),
        date=date,
        force=force,
        rejected_ids={q.id for q in rejected},
    )
    outcome.dry_run = dry_run
    if translations_path is not None:
        translations = load_for_review(Path(translations_path))
        if not translations:
            raise AuditError(f"no translated questions in {translations_path}")
        gone = {q.id for q in rejected} | {q.id for q in outcome.rejected}
        outcome.translations, outcome.translations_synced = sync_translations(
            translations, outcome.questions, rejected_ids=gone
        )
        outcome.translation_sources = {
            t.id: t.derived_from[0]
            for t in outcome.translations
            if t.id in outcome.translations_synced
        }
    if not dry_run:
        if outcome.rejected:
            save_questions([*rejected, *outcome.rejected], rejected_path)
        save_questions(outcome.questions, questions_path)
        if outcome.translations is not None and outcome.translations_synced:
            save_questions(outcome.translations, Path(translations_path))
    return outcome


def format_summary(outcome: AuditOutcome) -> str:
    """Counts per action, evidence and rejection counts, and the resulting type x split table."""
    lines = ["decisions per action: " + ", ".join(f"{a}: {outcome.counts[a]}" for a in ACTIONS)]
    if outcome.already_rejected:
        lines.append(f"reject decisions already applied earlier: {len(outcome.already_rejected)}")
    lines += [
        f"evidence pages removed: {outcome.evidence_removed}",
        f"derived evidence recomputed: {len(outcome.derived_recomputed)}"
        + (f" ({', '.join(outcome.derived_recomputed)})" if outcome.derived_recomputed else ""),
        f"rows moved to rejected: {len(outcome.rejected)}",
        f"audit status: {_audit_counts(outcome.questions)}",
        "",
        f"resulting file: {len(outcome.questions)} questions",
        f"{'type':<14}{'dev':>6}{'test':>6}{'total':>7}",
    ]
    cells = Counter((q.type, q.split) for q in outcome.questions)
    for t in QUESTION_TYPES:
        dev, test = cells[(t, "dev")], cells[(t, "test")]
        lines.append(f"{t:<14}{dev:>6}{test:>6}{dev + test:>7}")
    n = len(outcome.questions)
    dev = sum(q.split == "dev" for q in outcome.questions)
    lines.append(f"{'total':<14}{dev:>6}{n - dev:>6}{n:>7}")
    if outcome.translations is not None:
        pairs = [f"{t} <- {outcome.translation_sources[t]}" for t in outcome.translations_synced]
        lines.append(
            f"translations synced: {len(pairs)}" + (f" ({', '.join(pairs)})" if pairs else "")
        )
    if outcome.dry_run:
        lines.append("dry run: nothing written")
    return "\n".join(lines)


def _audit_counts(questions: list[Question]) -> str:
    counts = Counter(q.audit.status if q.audit else "none" for q in questions)
    audited = sum(q.audited for q in questions)
    parts = [f"{s}: {counts[s]}" for s in ("passed", "fixed", "flagged", "none")]
    return ", ".join(parts) + f" (audited subset: {audited})"
