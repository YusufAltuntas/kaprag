"""The router graph (LangGraph `StateGraph`, explicit nodes, no tool-calling).

    START -> classify --simple--> simple -> END
                      --multi---> decompose --fallback--> simple
                                            --Send per look-up--> lookup (concurrent)
             lookup -> compute --ok--> synthesize -> END
                               --missing/refused--> abstain -> END

- `classify` (Haiku, `RouteDecision`, step `classify`). A failed call -> `simple`.
- `simple`: the configured RAG pipeline unchanged (`RagPipeline.run`), so a simple
  question gets exactly that pipeline's answer (and its LLM cache entry).
- `decompose` (Sonnet, `Plan`, step `decompose`); the report-year rule is applied in
  code (`apply_report_year_rule`). A failed call, an inexpressible question or more
  look-ups than `max_lookups` -> `simple`, recorded as `fallback`.
- `lookup`, one per planned look-up via `Send`, run concurrently: the same retriever
  as the simple path with a SQL filter built in code from the look-up's bank and
  report year (`metadata_filter`, plus the access condition when a role applies),
  then the simple path's generation prompt and post-processing on the sub-question
  (`generate(..., step="sub_answer")`, `build_answer`). A failed call -> that look-up
  is missing.
- `compute`: `kaprag.agent.compute` on the normalised figures. No LLM.
- `synthesize` (Haiku, `SynthesizedAnswer`, step `synthesize`): answer text only;
  `value` / `unit` / `basis` come from `compute`, citations are the union of the
  look-ups' citations. A failed call -> a templated Turkish sentence.
- `abstain`: a missing figure or a refused computation -> an abstention naming it.

Every LLM call carries the runner's tags plus its `step` (and `lookup` id), so the
runner attributes its cost and latency to the question. Unexpected exceptions
(anything but `LLMCallError`) propagate, so the runner records the question as an
error. Debug output (`PipelineOutput.debug`): `route`, `router` = {classify,
decompose (plan, year-rule fixes), lookups (per look-up: filter, retrieved chunks,
answer, raw and normalised value, citations, error), compute (operation, inputs,
result), synthesize, fallback}, plus the simple pipeline's own debug on that route.
"""

from __future__ import annotations

import asyncio
import operator
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from kaprag.access.filter import access_filter
from kaprag.access.roles import effective_roles
from kaprag.agent.compute import (
    BASIS_UNITS,
    Computation,
    Operand,
    compute,
    format_number,
    format_quantity,
)
from kaprag.agent.plan import (
    CLASSIFY_SYSTEM_PROMPT,
    SYNTHESIZE_SYSTEM_PROMPT,
    Lookup,
    Plan,
    RouteDecision,
    SynthesizedAnswer,
    apply_report_year_rule,
    classify_user_prompt,
    decompose_system_prompt,
    decompose_user_prompt,
    synthesize_user_prompt,
)
from kaprag.corpus import BANK_NAMES
from kaprag.eval.answer import Citation, SystemAnswer
from kaprag.eval.pipeline import PipelineOutput, RetrievedItem
from kaprag.llm import LLMCallError, LLMClient
from kaprag.rag.context import format_context
from kaprag.rag.generate import generate
from kaprag.rag.pipeline import RagPipeline
from kaprag.rag.postprocess import build_answer
from kaprag.rag.query_filter import metadata_filter


@dataclass(frozen=True)
class RouterModels:
    classify: str
    decompose: str
    synthesize: str
    max_lookups: int


@dataclass
class LookupResult:
    """One look-up's outcome: its answer through the simple path's post-processing
    (None when the call failed), what was retrieved, and debug fields."""

    index: int
    lookup: Lookup
    answer: SystemAnswer | None
    retrieved: list[RetrievedItem]
    debug: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


class RouterState(TypedDict, total=False):
    question: str
    tags: dict[str, Any]
    access_role: str | None  # the caller's role (None: the config's, if any)
    role: str | None  # the effective role of the request
    route: str
    classify: dict[str, Any]
    decompose: dict[str, Any] | None
    fallback: str | None
    plan: Plan | None
    lookups: Annotated[list[LookupResult], operator.add]
    computation: Computation
    output: PipelineOutput


class LookupTask(TypedDict):
    """What `Send` hands one `lookup` node."""

    index: int
    lookup: Lookup
    question: str
    tags: dict[str, Any]
    role: str | None


def build_graph(rag: RagPipeline, models: RouterModels) -> Any:
    """Compile the router graph around the simple pipeline `rag` (whose retriever,
    retrieval params, generation model and LLM client the look-ups share)."""
    llm: Callable[[], LLMClient] = rag.llm
    config = rag.config

    async def classify_node(state: RouterState) -> dict[str, Any]:
        try:
            result = await llm().complete(
                CLASSIFY_SYSTEM_PROMPT,
                classify_user_prompt(state["question"]),
                models.classify,
                schema=RouteDecision,
                tags={**state["tags"], "step": "classify"},
            )
        except LLMCallError as exc:
            return {
                "route": "simple",
                "classify": {"route": None, "reason": None, "error": str(exc)},
                "fallback": "classify_error",
            }
        decision: RouteDecision = result.parsed
        return {
            "route": decision.route,
            "classify": {"route": decision.route, "reason": decision.reason, "error": None},
        }

    async def decompose_node(state: RouterState) -> dict[str, Any]:
        try:
            result = await llm().complete(
                decompose_system_prompt(models.max_lookups),
                decompose_user_prompt(state["question"]),
                models.decompose,
                schema=Plan,
                tags={**state["tags"], "step": "decompose"},
            )
        except LLMCallError as exc:
            return _fallback("decompose_error", {"plan": None, "error": str(exc)})
        plan: Plan = result.parsed
        if not plan.expressible:
            return _fallback("inexpressible", {"plan": _dump(plan), "error": None})
        if len(plan.lookups) > models.max_lookups:
            return _fallback("too_many_lookups", {"plan": _dump(plan), "error": None})
        plan, fixes = apply_report_year_rule(plan)
        return {
            "plan": plan,
            "decompose": {"plan": _dump(plan), "year_rule_fixes": fixes, "error": None},
        }

    def after_decompose(state: RouterState) -> str | list[Send]:
        if state["route"] == "simple":
            return "simple"
        plan = state["plan"]
        return [
            Send(
                "lookup",
                LookupTask(
                    index=i,
                    lookup=lookup,
                    question=state["question"],
                    tags=state["tags"],
                    role=state["role"],
                ),
            )
            for i, lookup in enumerate(plan.lookups)
        ]

    async def lookup_node(task: LookupTask) -> dict[str, Any]:
        lookup = task["lookup"]
        where = metadata_filter([lookup.bank], [lookup.report_year])
        if task["role"] is not None:
            where = access_filter(task["role"], where)
        ranked = await asyncio.to_thread(
            rag.retriever.retrieve, lookup.sub_question, config.retrieve_k, where=where
        )
        retrieved = [r.item for r in ranked]
        sources = retrieved[: config.context_k]
        debug: dict[str, Any] = {
            "id": lookup.id,
            "filter": {"banks": [lookup.bank], "years": [lookup.report_year]},
            "retrieved": [_item_summary(item) for item in retrieved],
            "context_chunk_ids": [s.chunk_id for s in sources],
        }
        try:
            generated = await generate(
                llm(),
                lookup.sub_question,
                format_context(sources),
                model=config.generation.model,
                tags={**task["tags"], "lookup": lookup.id},
                step="sub_answer",
            )
        except LLMCallError as exc:
            result = LookupResult(task["index"], lookup, None, retrieved, debug, str(exc))
            return {"lookups": [result]}
        answer, post_debug = build_answer(generated, sources)
        debug |= {"answer": answer.model_dump(mode="json"), **post_debug}
        return {"lookups": [LookupResult(task["index"], lookup, answer, retrieved, debug)]}

    def compute_node(state: RouterState) -> dict[str, Any]:
        plan = state["plan"]
        by_id = {r.lookup.id: r for r in state["lookups"]}
        operands = [_operand(by_id[i]) for i in plan.operation.operands]
        return {
            "computation": compute(
                plan.operation.kind, operands, descending=plan.operation.descending
            )
        }

    def after_compute(state: RouterState) -> str:
        return "synthesize" if state["computation"].ok else "abstain"

    async def synthesize_node(state: RouterState) -> dict[str, Any]:
        comp, plan = state["computation"], state["plan"]
        results = _ordered(state["lookups"])
        sentence = result_sentence(comp, plan, results)
        error = None
        try:
            reply = await llm().complete(
                SYNTHESIZE_SYSTEM_PROMPT,
                synthesize_user_prompt(state["question"], facts_block(plan, results, sentence)),
                models.synthesize,
                schema=SynthesizedAnswer,
                tags={**state["tags"], "step": "synthesize"},
            )
            text = reply.parsed.answer
        except LLMCallError as exc:
            error = str(exc)
            text = templated_answer(comp, plan, results, sentence)
        answer = SystemAnswer(
            answer=text,
            value=comp.value,
            unit=comp.unit,
            basis=comp.basis,
            citations=union_citations(r.answer for r in results),
        )
        synth = {"error": error, "templated": error is not None, "result_sentence": sentence}
        return {"output": _multi_output(state, answer, results, synth)}

    def abstain_node(state: RouterState) -> dict[str, Any]:
        comp, results = state["computation"], _ordered(state["lookups"])
        if comp.missing:
            operands = [_operand(r) for r in results]
            missing = "; ".join(f"{o.label} ({o.missing})" for o in operands if o.missing)
            text = (
                "Hesaplama için gereken şu rakamlar elde edilemedi: "
                f"{missing}. Bu yüzden soruyu cevaplayamıyorum."
            )
        else:
            text = f"Rakamlar bulundu ancak hesaplama yapılamadı: {comp.refusal}."
        answer = SystemAnswer(answer=text, abstained=True)
        return {"output": _multi_output(state, answer, results, None)}

    async def simple_node(state: RouterState) -> dict[str, Any]:
        output = await rag.run(
            state["question"], tags=state["tags"], access_role=state["access_role"]
        )
        router = _router_debug(state)
        return {
            "output": output.model_copy(
                update={"debug": {"route": "simple", "router": router, **output.debug}}
            )
        }

    def _multi_output(
        state: RouterState,
        answer: SystemAnswer,
        results: list[LookupResult],
        synth: dict[str, Any] | None,
    ) -> PipelineOutput:
        router = _router_debug(state)
        plan, comp = state["plan"], state["computation"]
        by_id = {r.lookup.id: r for r in results}
        router["lookups"] = [{**r.debug, "error": r.error} for r in results]
        router["compute"] = {
            "operation": plan.operation.model_dump(mode="json"),
            "inputs": [_operand_debug(_operand(by_id[i])) for i in plan.operation.operands],
            "result": comp.to_debug(),
        }
        router["synthesize"] = synth
        debug: dict[str, Any] = {"route": "multi", "router": router}
        if state["role"] is not None:
            debug["access"] = {
                "role": state["role"],
                "effective_roles": list(effective_roles(state["role"])),
                "source": "caller" if state["access_role"] is not None else "config",
            }
        retrieved = interleave([r.retrieved for r in results], config.retrieve_k)
        return PipelineOutput(answer=answer, retrieved=retrieved, debug=debug)

    graph = StateGraph(RouterState)
    graph.add_node("classify", classify_node)
    graph.add_node("simple", simple_node)
    graph.add_node("decompose", decompose_node)
    graph.add_node("lookup", lookup_node)
    graph.add_node("compute", compute_node)
    graph.add_node("synthesize", synthesize_node)
    graph.add_node("abstain", abstain_node)
    graph.add_edge(START, "classify")
    graph.add_conditional_edges(
        "classify",
        lambda s: "simple" if s["route"] == "simple" else "decompose",
        ["simple", "decompose"],
    )
    graph.add_conditional_edges("decompose", after_decompose, ["simple", "lookup"])
    graph.add_edge("lookup", "compute")
    graph.add_conditional_edges("compute", after_compute, ["synthesize", "abstain"])
    graph.add_edge("simple", END)
    graph.add_edge("synthesize", END)
    graph.add_edge("abstain", END)
    return graph.compile()


# --- helpers (pure) -----------------------------------------------------------------


def _fallback(reason: str, decompose: dict[str, Any]) -> dict[str, Any]:
    return {"route": "simple", "fallback": reason, "decompose": decompose, "plan": None}


def _dump(plan: Plan) -> dict[str, Any]:
    return plan.model_dump(mode="json")


def _router_debug(state: RouterState) -> dict[str, Any]:
    return {
        "classify": state.get("classify"),
        "decompose": state.get("decompose"),
        "fallback": state.get("fallback"),
    }


def _item_summary(item: RetrievedItem) -> dict[str, Any]:
    return {
        "chunk_id": item.chunk_id,
        "doc": item.doc,
        "page_start": item.page_start,
        "page_end": item.page_end,
    }


def _ordered(results: Sequence[LookupResult]) -> list[LookupResult]:
    return sorted(results, key=lambda r: r.index)


def lookup_label(lookup: Lookup) -> str:
    """`Akbank 2024 şube sayısı`: names a figure for the user."""
    return f"{BANK_NAMES[lookup.bank]} {lookup.fiscal_year} {lookup.metric}"


def _operand(result: LookupResult) -> Operand:
    """A look-up as a `compute` operand. The basis is the answer's, else the one the
    plan asked for (the sub-question stated it)."""
    lookup, answer = result.lookup, result.answer
    label = lookup_label(lookup)
    if answer is None:
        return Operand(lookup.id, label, None, None, None, missing="alt soru cevaplanamadı")
    if answer.abstained:
        return Operand(lookup.id, label, None, None, None, missing="kaynaklarda bulunamadı")
    if answer.clarification:
        return Operand(lookup.id, label, None, None, None, missing="netleştirme istendi")
    if answer.value is None:
        return Operand(lookup.id, label, None, None, None, missing="sayısal değer çıkarılamadı")
    return Operand(lookup.id, label, answer.value, answer.unit, answer.basis or lookup.basis)


def _operand_debug(operand: Operand) -> dict[str, Any]:
    return {
        "id": operand.id,
        "label": operand.label,
        "value": operand.value,
        "unit": operand.unit,
        "basis": operand.basis,
        "missing": operand.missing,
    }


def interleave(lists: Sequence[Sequence[RetrievedItem]], k: int) -> list[RetrievedItem]:
    """Round-robin over ranked lists (rank 1 of each list, then rank 2, ...), skipping
    chunks already taken; the first `k`."""
    merged: list[RetrievedItem] = []
    seen: set[str] = set()
    for rank in range(max((len(x) for x in lists), default=0)):
        for ranked in lists:
            if rank < len(ranked) and ranked[rank].chunk_id not in seen:
                seen.add(ranked[rank].chunk_id)
                merged.append(ranked[rank])
    return merged[:k]


def union_citations(answers: Sequence[SystemAnswer | None] | Any) -> list[Citation]:
    """The look-ups' citations in look-up order, de-duplicated."""
    citations: list[Citation] = []
    for answer in answers:
        for citation in answer.citations if answer is not None else []:
            if citation not in citations:
                citations.append(citation)
    return citations


def result_sentence(comp: Computation, plan: Plan, results: Sequence[LookupResult]) -> str:
    """The computed result in one Turkish sentence (no LLM)."""
    labels = {r.lookup.id: lookup_label(r.lookup) for r in results}
    values = {r.lookup.id: r.answer.value if r.answer else None for r in results}
    ids = plan.operation.operands
    value = comp.value
    if comp.kind == "percent_change":
        base, target = (labels[i] for i in ids)
        if value == 0:
            return f"{target}, {base} değerine göre değişmedi (%0)."
        direction = "artış" if value > 0 else "azalış"
        change = format_quantity(abs(value), "%")
        return f"{target}, {base} değerine göre {change} {direction} gösterdi."
    if comp.kind == "difference":
        a, b = (labels[i] for i in ids)
        shown = (
            f"{format_number(value)} yüzde puan"
            if comp.unit == "%"
            else format_quantity(value, comp.unit)
        )
        return f"{a} ile {b} arasındaki fark ({a} eksi {b}): {shown}."
    if comp.kind == "ratio":
        a, b = (labels[i] for i in ids)
        return f"{a}, {b} değerinin {format_number(value)} katıdır."
    if comp.kind == "share":
        a, b = (labels[i] for i in ids)
        return f"{a}, {b} değerinin {format_quantity(value, '%')} kadarıdır."
    if comp.kind in ("max", "min"):
        extreme = "en yüksek" if comp.kind == "max" else "en düşük"
        winners = " ve ".join(labels[i] for i in comp.winners)
        shown = format_quantity(value, comp.unit)
        return f"Karşılaştırılanlar arasında {extreme} değer: {winners} ({shown})."
    if comp.kind == "rank":
        direction = "büyükten küçüğe" if plan.operation.descending else "küçükten büyüğe"
        order = "; ".join(
            f"{labels[i]}: {format_quantity(values[i], comp.unit)}" for i in comp.order
        )
        return f"Sıralama ({direction}): {order}."
    raise ValueError(f"unknown operation {comp.kind!r}")


def _figures(results: Sequence[LookupResult]) -> list[str]:
    """`label: figure (report, basis)` per look-up with a value; the basis only for
    money and ratios, where `compute` checks it (a count has none)."""
    lines = []
    for r in results:
        operand = _operand(r)
        if operand.missing:
            continue
        show_basis = operand.basis and operand.unit in BASIS_UNITS
        basis = f", {operand.basis}" if show_basis else ""
        lines.append(
            f"{operand.label}: {format_quantity(operand.value, operand.unit)} "
            f"({r.lookup.report_year} raporu{basis})"
        )
    return lines


def facts_block(plan: Plan, results: Sequence[LookupResult], sentence: str) -> str:
    """What the synthesis model sees besides the question."""
    figures = "\n".join(f"- {line}" for line in _figures(results))
    return f"Kaynaklardan bulunan rakamlar:\n{figures}\n\nKodla yapılan hesabın sonucu: {sentence}"


def templated_answer(
    comp: Computation, plan: Plan, results: Sequence[LookupResult], sentence: str
) -> str:
    """The answer when synthesis fails: the computed result and its inputs, plainly."""
    text = f"{sentence} Kullanılan rakamlar: {'; '.join(_figures(results))}."
    if comp.basis is not None:
        text += f" Rakamlar {comp.basis} esaslıdır."
    return text
