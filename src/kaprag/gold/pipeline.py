"""Runs the gold drafting stages in order (eval/README.md §7 steps 1-7, 9) for one run folder.

Every stage writes its artefact to the run folder; LLM replies are disk-cached by
`kaprag.llm`, so re-running the same configuration is resumable and replays finished
calls for free. A failed call or a rejected draft never aborts the run: it is logged
(`failures.jsonl`, `drops.jsonl`) and counted in `summary.json`.
"""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kaprag.corpus import Document
from kaprag.eval.schema import save_questions
from kaprag.gold import abstain, ambiguous, checking, drafting, probes
from kaprag.gold.assemble import DEFAULT_QUOTAS, PILOT_SIZE, assemble
from kaprag.gold.derive import compose_from_probes
from kaprag.gold.draft import Draft, write_drafts
from kaprag.gold.figures import FigureError
from kaprag.gold.metrics import METRICS, Metric
from kaprag.gold.pages import SEED, DocPages, sample_pages
from kaprag.gold.runlog import RunLog, call_llm
from kaprag.gold.search import Searcher
from kaprag.parsing.outputs import DEFAULT_PARSED_DIR, write_jsonl_atomic


@dataclass
class RunConfig:
    run_dir: Path
    out: Path
    docs: Sequence[Document]
    table_pages: int = 4
    text_pages: int = 4
    table_requests: dict[str, int] = field(default_factory=lambda: {"tablo": 2, "hesaplama": 1})
    text_requests: dict[str, int] = field(default_factory=lambda: {"olgusal": 2, "sentez": 1})
    metrics: Sequence[Metric] = METRICS
    abstain_per_kind: int = 18
    abstain_batch: int = 4
    ambiguous_per_kind: int = 9
    quotas: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_QUOTAS))
    pilot: int = PILOT_SIZE
    seed: int = SEED
    parsed_dir: Path = DEFAULT_PARSED_DIR


class Pipeline:
    def __init__(
        self, cfg: RunConfig, client: Any, searcher_factory: Callable[[], Searcher]
    ) -> None:
        self.cfg = cfg
        self.client = client
        self._searcher_factory = searcher_factory
        self._searcher: Searcher | None = None
        self.log = RunLog()
        self._pages: dict[str, DocPages] = {}
        self.counts: dict[str, Any] = {}

    # --- helpers -----------------------------------------------------------------------

    @property
    def searcher(self) -> Searcher:
        if self._searcher is None:
            self._searcher = self._searcher_factory()
        return self._searcher

    def pages(self, doc_id: str) -> DocPages:
        if doc_id not in self._pages:
            self._pages[doc_id] = DocPages.load(doc_id, self.cfg.parsed_dir)
        return self._pages[doc_id]

    def _path(self, name: str) -> Path:
        return self.cfg.run_dir / name

    # --- step 1: sampling --------------------------------------------------------------

    def sample(self) -> list[drafting.PageTask]:
        tasks: list[drafting.PageTask] = []
        rows: list[dict] = []
        for doc in self.cfg.docs:
            infos = self.pages(doc.doc_id).infos(doc.year)
            for info in sample_pages(
                infos, self.cfg.table_pages, self.cfg.text_pages, self.cfg.seed
            ):
                requests = (
                    self.cfg.table_requests if info.stratum == "table" else self.cfg.text_requests
                )
                requests = {
                    t: n for t, n in requests.items() if t != "hesaplama" or info.has_comparative
                }
                tasks.append(drafting.PageTask(doc.doc_id, doc.bank, doc.year, info.page, requests))
                rows.append(info.to_dict() | {"requests": requests})
        write_jsonl_atomic(self._path("pages.jsonl"), rows)
        self.counts["pages_sampled"] = len(tasks)
        return tasks

    # --- steps 2-3: page drafts and their check -------------------------------------------

    async def _page(self, task: drafting.PageTask) -> tuple[list[Draft], list[Draft]]:
        docpages = self.pages(task.doc_id)
        text = docpages.render(task.page)
        result = await call_llm(
            self.client,
            self.log,
            stage="draft",
            key=task.key,
            system=drafting.SYSTEM,
            user=drafting.build_prompt(task, text),
            schema=drafting.PageDrafts,
        )
        if result is None:
            return [], []
        drafts, drops = drafting.validate_page(result, task, text, docpages.numbers(task.page))
        for key, reason, out in drops:
            self.log.drop("draft_validation", key, reason, out)
        if not drafts:
            return [], []
        check = await call_llm(
            self.client,
            self.log,
            stage="check",
            key=task.key,
            system=checking.SYSTEM,
            user=checking.build_prompt(task.bank, task.year, [(task.page, text)], drafts),
            schema=checking.PageCheck,
        )
        passed, failed = checking.apply_check(drafts, check)
        for d, reason in failed:
            self.log.drop("check", d.key, reason, d)
        return drafts, passed

    async def draft_pages(self, tasks: Sequence[drafting.PageTask]) -> list[Draft]:
        results = await asyncio.gather(*(self._page(t) for t in tasks))
        drafted = [d for ds, _ in results for d in ds]
        checked = [d for _, ps in results for d in ps]
        write_drafts(self._path("page_drafts.jsonl"), drafted)
        write_drafts(self._path("page_checked.jsonl"), checked)
        self.counts["page_drafts_valid"] = len(drafted)
        self.counts["page_drafts_checked"] = len(checked)
        return checked

    # --- step 4: probes and derived questions -------------------------------------------

    def _retrieve(self, doc: Document) -> list:
        hits = {
            m.id: self.searcher.search(
                probes.metric_query(m, doc.bank, doc.year),
                [doc.doc_id],
                k=probes.PASSAGES_PER_METRIC,
            )
            for m in self.cfg.metrics
        }
        return probes.collect_passages(hits)

    async def _probe_doc(self, doc: Document, passages: list) -> list[Draft]:
        key = f"probe:{doc.doc_id}"
        if not passages:
            self.log.fail("probe_extract", key, "no passages retrieved")
            return []
        result = await call_llm(
            self.client,
            self.log,
            stage="probe_extract",
            key=key,
            system=probes.SYSTEM,
            user=probes.build_prompt(doc.bank, doc.year, self.cfg.metrics, passages),
            schema=probes.ProbeExtraction,
        )
        if result is None:
            return []
        docpages = self.pages(doc.doc_id)
        outs = {o.metric: o for o in result.items}
        drafts: list[Draft] = []
        for metric in self.cfg.metrics:
            mkey = f"{key}:{metric.id}"
            out = outs.get(metric.id)
            if out is None:
                self.log.drop("probe_validation", mkey, "metric missing from reply")
                continue
            try:
                drafts.append(
                    probes.validate_probe(
                        out, metric, doc.doc_id, doc.bank, doc.year, passages, docpages
                    )
                )
            except (FigureError, ValueError) as exc:
                self.log.drop("probe_validation", mkey, str(exc), out)
        if not drafts:
            return []
        pages = sorted({d.evidence[0].page for d in drafts})
        check = await call_llm(
            self.client,
            self.log,
            stage="probe_check",
            key=key,
            system=checking.SYSTEM,
            user=checking.build_prompt(
                doc.bank, doc.year, [(p, docpages.render(p)) for p in pages], drafts
            ),
            schema=checking.PageCheck,
        )
        passed, failed = checking.apply_check(drafts, check)
        for d, reason in failed:
            self.log.drop("probe_check", d.key, reason, d)
        return passed

    async def probe(self) -> tuple[list[Draft], list[Draft]]:
        passages = {doc.doc_id: self._retrieve(doc) for doc in self.cfg.docs}
        write_jsonl_atomic(
            self._path("probe_passages.jsonl"),
            [
                {"doc_id": d, "chunks": [[h.chunk_id, h.page_start, h.page_end] for h in hits]}
                for d, hits in passages.items()
            ],
        )
        results = await asyncio.gather(
            *(self._probe_doc(doc, passages[doc.doc_id]) for doc in self.cfg.docs)
        )
        checked = [d for ds in results for d in ds]
        derived = compose_from_probes(checked)
        write_drafts(self._path("probes.jsonl"), checked)
        write_drafts(self._path("derived.jsonl"), derived)
        self.counts["probes_checked"] = len(checked)
        self.counts["derived_candidates"] = dict(Counter(d.type for d in derived))
        return checked, derived

    # --- step 5: cevapsiz ---------------------------------------------------------------

    async def _judge(self, batch: list[tuple[str, abstain.AbstainOut, list]]) -> list[Draft]:
        key = f"abstain_judge:{batch[0][0]}"
        result = await call_llm(
            self.client,
            self.log,
            stage="abstain_judge",
            key=key,
            system=abstain.JUDGE_SYSTEM,
            user=abstain.build_judge_prompt([(o, h) for _, o, h in batch]),
            schema=abstain.AbstainJudgement,
        )
        verdicts = {v.index: v for v in result.items} if result is not None else {}
        kept: list[Draft] = []
        for i, (ckey, out, hits) in enumerate(batch):
            verdict = verdicts.get(i)
            reason = abstain.keep_reason(out, verdict) if result is not None else "judge failed"
            if reason is not None:
                self.log.drop("abstain_judge", ckey, reason, out)
                continue
            kept.append(abstain.to_draft(ckey, out, verdict, hits))
        return kept

    async def abstain(self, facts: Sequence[Draft]) -> list[Draft]:
        n = self.cfg.abstain_per_kind
        if n <= 0:
            return []
        result = await call_llm(
            self.client,
            self.log,
            stage="abstain_draft",
            key="abstain",
            system=abstain.DRAFT_SYSTEM,
            user=abstain.build_draft_prompt(n, facts),
            schema=abstain.AbstainDrafts,
        )
        if result is None:
            return []
        candidates: list[tuple[str, abstain.AbstainOut, list]] = []
        for i, out in enumerate(result.questions):
            ckey = f"abstain:{i}"
            try:
                abstain.validate_candidate(out)
            except ValueError as exc:
                self.log.drop("abstain_validation", ckey, str(exc), out)
                continue
            hits = self.searcher.search(out.question, abstain.doc_filter(out), k=abstain.PASSAGES)
            candidates.append((ckey, out, hits))
        size = max(1, self.cfg.abstain_batch)
        batches = [candidates[i : i + size] for i in range(0, len(candidates), size)]
        results = await asyncio.gather(*(self._judge(b) for b in batches))
        kept = [d for ds in results for d in ds]
        write_drafts(self._path("abstain.jsonl"), kept)
        self.counts["abstain_candidates"] = len(result.questions)
        self.counts["abstain_kept"] = len(kept)
        return kept

    # --- step 6: belirsiz ----------------------------------------------------------------

    async def ambiguous(self, singles: Sequence[Draft]) -> list[Draft]:
        tasks = ambiguous.select_sources(singles, self.cfg.ambiguous_per_kind, self.cfg.seed)
        if not tasks:
            return []
        result = await call_llm(
            self.client,
            self.log,
            stage="ambiguous",
            key="ambiguous",
            system=ambiguous.SYSTEM,
            user=ambiguous.build_prompt(tasks),
            schema=ambiguous.Rewrites,
        )
        if result is None:
            return []
        rewrites = {r.index: r.question for r in result.items}
        kept: list[Draft] = []
        for i, task in enumerate(tasks):
            key = f"ambiguous:{task.kind}:{task.source.key}"
            if i not in rewrites:
                self.log.drop("ambiguous", key, "no rewrite returned")
                continue
            try:
                kept.append(ambiguous.to_draft(task, rewrites[i]))
            except ValueError as exc:
                self.log.drop("ambiguous", key, str(exc), {"question": rewrites[i]})
        write_drafts(self._path("ambiguous.jsonl"), kept)
        self.counts["ambiguous_kept"] = len(kept)
        return kept

    # --- all ---------------------------------------------------------------------------

    async def run(self) -> dict[str, Any]:
        self.cfg.run_dir.mkdir(parents=True, exist_ok=True)
        tasks = self.sample()
        page_singles = await self.draft_pages(tasks)
        probe_singles, derived = await self.probe()
        abstained = await self.abstain(probe_singles)
        ambiguous_drafts = await self.ambiguous(probe_singles + page_singles)
        pool = page_singles + probe_singles + derived + abstained + ambiguous_drafts
        questions = assemble(pool, self.cfg.quotas, self.cfg.pilot, self.cfg.seed)
        save_questions(questions, self.cfg.out)
        self.counts["questions"] = dict(Counter(q.type for q in questions))
        self.counts["drops_by_stage"] = dict(Counter(d["stage"] for d in self.log.drops))
        self.counts["failures_by_stage"] = dict(Counter(f["stage"] for f in self.log.failures))
        self.log.write(self.cfg.run_dir)
        summary = {"counts": self.counts, "out": str(self.cfg.out)}
        self._path("summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return summary


def trace_summary(trace_path: Path) -> dict[str, Any]:
    """Calls, cache hits, errors and API-equivalent cost per model from a trace file."""
    per_model: dict[str, Counter] = {}
    cost: dict[str, float] = {}
    if not trace_path.is_file():
        return {}
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        c = per_model.setdefault(r["model"], Counter())
        c["calls"] += 1
        c["cached"] += bool(r["cached"])
        c["errors"] += r["error"] is not None
        if not r["cached"]:
            cost[r["model"]] = cost.get(r["model"], 0.0) + (r["cost_usd"] or 0.0)
    return {
        m: dict(c) | {"cost_usd_uncached": round(cost.get(m, 0.0), 4)} for m, c in per_model.items()
    }
