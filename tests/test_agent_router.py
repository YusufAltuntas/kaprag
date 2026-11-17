"""The router graph end to end: the real LangGraph graph with a fake LLM backend and a
fake retriever (no GPU, DB or network)."""

import asyncio
import json
import threading

import pytest

from kaprag.agent import plan as plan_module
from kaprag.agent.graph import interleave
from kaprag.agent.pipeline import RouterPipeline, parse_router_config
from kaprag.eval.answer import Citation
from kaprag.eval.config import load_config
from kaprag.eval.pipeline import RetrievedItem, resolve_factory
from kaprag.llm import BackendError, LLMClient, Usage
from kaprag.llm.types import BackendResponse
from kaprag.rag import generate as generate_module
from kaprag.rag.retriever import RankedChunk
from kaprag.retrieval import SqlFilter

TAGS = {"run": "r1", "qid": "q101", "role": "pipeline"}
RAG = {"index": "fixed", "retrieve_k": 10, "context_k": 5}


# --- fakes ---------------------------------------------------------------------------


def item(doc, n, query_no):
    return RetrievedItem(
        chunk_id=f"{doc}:fixed:{query_no}-{n}",
        doc=doc,
        page_start=10 + n,
        page_end=10 + n,
        score=1.0 - n / 100,
        text=f"{doc} chunk {n}",
    )


class FakeRetriever:
    """Returns 12 chunks of the filtered bank's report for the filtered year (or
    `akbank-2024` unfiltered); with `shared`, every query's first chunk is the same."""

    def __init__(self, shared=False):
        self.calls = []
        self.shared = shared
        self.closed = False
        self.lock = threading.Lock()

    def retrieve(self, question, k, where=None):
        with self.lock:
            self.calls.append({"question": question, "k": k, "where": where})
            query_no = len(self.calls)
        doc = "akbank-2024"
        if isinstance(where, SqlFilter) and where:
            params = dict(where.params)
            doc = f"{params['f0'][0]}-{params['f1'][0]}"
        items = [item(doc, n, query_no) for n in range(1, 13)]
        if self.shared:
            items[0] = item("shared-2024", 0, 0)
        return [RankedChunk(i) for i in items[:k]]

    def close(self):
        self.closed = True


class Backend:
    """Answers by step, recognised from the system prompt; `sub` maps a sub-question
    to its reply. A reply that is an Exception is raised."""

    name = "fake"

    def __init__(self, *, route="multi", plan=None, sub=None, synth="Sentez metni.", simple=None):
        self.replies = {
            "classify": route,
            "decompose": plan,
            "synthesize": synth,
            "generate": simple,
        }
        self.sub = sub or {}
        self.calls = []
        self.in_flight = 0
        self.max_in_flight = 0

    async def generate(self, system, user, model):
        step = self.step(system)
        self.calls.append((step, model, user))
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            await asyncio.sleep(0.01)
            reply = self.reply(step, user)
        finally:
            self.in_flight -= 1
        if isinstance(reply, Exception):
            raise reply
        return BackendResponse(text=reply, usage=Usage(1000, 100))

    @staticmethod
    def step(system):
        if system == plan_module.CLASSIFY_SYSTEM_PROMPT:
            return "classify"
        if system.startswith(plan_module.decompose_system_prompt(8)[:60]):
            return "decompose"
        if system == plan_module.SYNTHESIZE_SYSTEM_PROMPT:
            return "synthesize"
        assert system == generate_module.SYSTEM_PROMPT
        return "generate"

    def reply(self, step, user):
        if step == "classify":
            route = self.replies["classify"]
            return route if isinstance(route, Exception) else json.dumps({"route": route})
        if step == "decompose":
            plan = self.replies["decompose"]
            return plan if isinstance(plan, Exception) else json.dumps(plan)
        if step == "synthesize":
            synth = self.replies["synthesize"]
            return synth if isinstance(synth, Exception) else json.dumps({"answer": synth})
        question = user.split("\n", 1)[0].removeprefix("Soru: ")
        if question in self.sub:
            reply = self.sub[question]
            return reply if isinstance(reply, Exception) else json.dumps(reply)
        return json.dumps(self.replies["generate"] or SIMPLE_REPLY)


SIMPLE_REPLY = {
    "answer": "Akbank'ın 2024 sonu şube sayısı 694'tür [1].",
    "value": "694",
    "unit": "şube",
    "citations": [1],
}


def lookup(id_, bank, year, question, report_year=None, basis=None):
    return {
        "id": id_,
        "bank": bank,
        "fiscal_year": year,
        "report_year": report_year or year,
        "metric": "şube sayısı",
        "basis": basis,
        "sub_question": question,
    }


def sub_reply(value, unit="adet", citations=(1,), basis=None, abstained=False):
    return {
        "answer": f"{value} {unit} [1].",
        "value": None if abstained else value,
        "unit": None if abstained else unit,
        "basis": basis,
        "citations": [] if abstained else list(citations),
        "abstained": abstained,
    }


QA = "Akbank'ın 2024 sonu şube sayısı?"
QG = "Garanti BBVA'nın 2024 sonu şube sayısı?"
MAX_PLAN = {
    "expressible": True,
    "lookups": [lookup("a", "akbank", 2024, QA), lookup("g", "garanti", 2024, QG)],
    "operation": {"kind": "max", "operands": ["a", "g"]},
}
MAX_SUB = {QA: sub_reply("694", citations=(1, 2)), QG: sub_reply("794", citations=(1,))}


def make(tmp_dir, backend, *, params=None, retriever=None):
    client = LLMClient(backend, cache_dir=None, trace_path=tmp_dir / "trace.jsonl")
    config = parse_router_config(params or {"rag": RAG})
    retriever = retriever or FakeRetriever()
    return RouterPipeline(config, retriever, lambda: client), retriever


def run(pipeline, question="Hangisinin şube sayısı daha yüksekti?", **kwargs):
    return asyncio.run(pipeline.run(question, tags=dict(TAGS), **kwargs))


def trace_rows(tmp_dir):
    text = (tmp_dir / "trace.jsonl").read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines()]


def steps(backend):
    return [s for s, _, _ in backend.calls]


# --- routing -------------------------------------------------------------------------


def test_simple_route_is_the_rag_pipeline(tmp_dir):
    backend = Backend(route="simple")
    pipeline, retriever = make(tmp_dir, backend)
    output = run(pipeline, "Akbank 2024 şube sayısı?")

    assert steps(backend) == ["classify", "generate"]
    assert [c["where"] for c in retriever.calls] == [None]
    assert retriever.calls[0]["question"] == "Akbank 2024 şube sayısı?"
    assert (output.answer.value, output.answer.unit) == (694, "adet")
    assert output.debug["route"] == "simple"
    assert output.debug["router"]["classify"]["route"] == "simple"
    assert output.debug["router"]["fallback"] is None
    assert output.debug["context_chunk_ids"]  # the simple pipeline's own debug is kept


def test_simple_route_reproduces_the_rag_pipeline_exactly(tmp_dir):
    from kaprag.rag.config import parse_rag_config
    from kaprag.rag.pipeline import RagPipeline

    backend = Backend(route="simple")
    pipeline, _ = make(tmp_dir, backend)
    routed = run(pipeline, "Akbank 2024 şube sayısı?")

    rag_backend = Backend()
    client = LLMClient(rag_backend, cache_dir=None, trace_path=tmp_dir / "rag.jsonl")
    rag = RagPipeline(parse_rag_config(RAG), FakeRetriever(), lambda: client)
    direct = asyncio.run(rag.run("Akbank 2024 şube sayısı?", tags=dict(TAGS)))

    assert routed.answer == direct.answer
    assert routed.retrieved == direct.retrieved
    assert backend.calls[1][1:] == rag_backend.calls[0][1:]  # same model and user prompt
    assert {k: v for k, v in routed.debug.items() if k not in ("route", "router")} == direct.debug


def test_multi_route_runs_the_whole_graph(tmp_dir):
    backend = Backend(plan=MAX_PLAN, sub=MAX_SUB, synth="Garanti BBVA'nın şubesi daha fazla.")
    pipeline, retriever = make(tmp_dir, backend)
    output = run(pipeline)

    assert sorted(steps(backend)) == sorted(
        ["classify", "decompose", "generate", "generate", "synthesize"]
    )
    assert output.debug["route"] == "multi"
    answer = output.answer
    assert answer.answer == "Garanti BBVA'nın şubesi daha fazla."
    assert (answer.value, answer.unit, answer.abstained) == (794, "adet", False)
    assert output.debug["router"]["compute"]["result"]["winners"] == ["g"]
    assert len(retriever.calls) == 2


def test_classify_failure_routes_simple_and_is_recorded(tmp_dir):
    backend = Backend(route=BackendError("boom"))
    pipeline, _ = make(tmp_dir, backend)
    output = run(pipeline)
    assert steps(backend) == ["classify", "generate"]
    assert output.debug["route"] == "simple"
    assert output.debug["router"]["fallback"] == "classify_error"
    assert "boom" in output.debug["router"]["classify"]["error"]


@pytest.mark.parametrize(
    ("plan", "fallback"),
    [
        (BackendError("down"), "decompose_error"),
        ({"expressible": False, "reason": "ortalama işlemi yok"}, "inexpressible"),
    ],
)
def test_decompose_failure_or_inexpressible_falls_back_to_simple(tmp_dir, plan, fallback):
    backend = Backend(plan=plan)
    pipeline, retriever = make(tmp_dir, backend)
    output = run(pipeline)
    assert steps(backend) == ["classify", "decompose", "generate"]
    assert output.debug["route"] == "simple"
    assert output.debug["router"]["fallback"] == fallback
    assert [c["where"] for c in retriever.calls] == [None]
    assert (output.answer.value, output.answer.unit) == (694, "adet")


def test_invalid_plan_twice_falls_back_to_simple(tmp_dir):
    bad = {**MAX_PLAN, "operation": {"kind": "average", "operands": ["a", "g"]}}
    backend = Backend(plan=bad)
    pipeline, _ = make(tmp_dir, backend)
    output = run(pipeline)
    assert steps(backend) == ["classify", "decompose", "decompose", "generate"]
    assert output.debug["router"]["fallback"] == "decompose_error"


def test_plan_over_the_configured_cap_falls_back_to_simple(tmp_dir):
    plan = {
        "expressible": True,
        "lookups": [lookup(x, "akbank", 2024, f"soru {x}") for x in "abc"],
        "operation": {"kind": "max", "operands": list("abc")},
    }
    backend = Backend(plan=plan)
    params = {"rag": RAG, "decompose": {"model": "claude-sonnet-5-5", "max_lookups": 2}}
    pipeline, _ = make(tmp_dir, backend, params=params)
    output = run(pipeline)
    assert output.debug["router"]["fallback"] == "too_many_lookups"
    assert steps(backend) == ["classify", "decompose", "generate"]


# --- look-ups ------------------------------------------------------------------------


def test_lookups_use_the_bank_and_report_year_filter_and_the_same_retriever(tmp_dir):
    backend = Backend(plan=MAX_PLAN, sub=MAX_SUB)
    pipeline, retriever = make(tmp_dir, backend)
    run(pipeline)

    calls = sorted(retriever.calls, key=lambda c: c["question"])
    assert [c["question"] for c in calls] == sorted([QA, QG])
    for call in calls:
        assert call["k"] == 10
        assert isinstance(call["where"], SqlFilter)
    params = sorted(tuple(map(tuple, c["where"].params.values())) for c in calls)
    assert params == [(("akbank",), (2024,)), (("garanti",), (2024,))]
    assert pipeline.rag.retriever is retriever


def test_lookup_reads_both_years_of_a_change_from_the_later_report(tmp_dir):
    q24, q25 = "Yapı Kredi 2024 mevduat?", "Yapı Kredi 2025 mevduat?"
    plan = {
        "expressible": True,
        "lookups": [
            lookup("y24", "yapikredi", 2024, q24, report_year=2024, basis="konsolide"),
            lookup("y25", "yapikredi", 2025, q25, report_year=2025, basis="konsolide"),
        ],
        "operation": {"kind": "percent_change", "operands": ["y24", "y25"]},
    }
    sub = {
        q24: sub_reply("531.701.291", "bin TL", basis="konsolide"),
        q25: sub_reply("615.217.176", "bin TL", basis="konsolide"),
    }
    backend = Backend(plan=plan, sub=sub)
    pipeline, retriever = make(tmp_dir, backend)
    output = run(pipeline)

    years = sorted(c["where"].params["f1"][0] for c in retriever.calls)
    assert years == [2025, 2025]
    assert output.debug["router"]["decompose"]["year_rule_fixes"] == [
        {"id": "y24", "from": 2024, "to": 2025}
    ]
    assert (output.answer.value, output.answer.unit, output.answer.basis) == (
        15.71,
        "%",
        "konsolide",
    )


def test_lookups_run_concurrently(tmp_dir):
    backend = Backend(plan=MAX_PLAN, sub=MAX_SUB)
    pipeline, _ = make(tmp_dir, backend)
    run(pipeline)
    assert backend.max_in_flight == 2


def test_lookup_failure_is_an_abstention_naming_it(tmp_dir):
    sub = {QA: BackendError("down"), QG: sub_reply("794")}
    backend = Backend(plan=MAX_PLAN, sub=sub)
    pipeline, _ = make(tmp_dir, backend)
    output = run(pipeline)

    assert "synthesize" not in steps(backend)
    answer = output.answer
    assert answer.abstained and answer.value is None and answer.citations == []
    assert "Akbank 2024 şube sayısı (alt soru cevaplanamadı)" in answer.answer
    assert "Garanti" not in answer.answer
    lookups = {lk["id"]: lk for lk in output.debug["router"]["lookups"]}
    assert "down" in lookups["a"]["error"]
    assert output.debug["router"]["compute"]["result"]["missing"] == ["Akbank 2024 şube sayısı"]


def test_abstaining_lookup_is_an_abstention_naming_it(tmp_dir):
    sub = {QA: sub_reply("694"), QG: sub_reply(None, abstained=True)}
    backend = Backend(plan=MAX_PLAN, sub=sub)
    pipeline, _ = make(tmp_dir, backend)
    output = run(pipeline)
    assert output.answer.abstained
    assert "Garanti BBVA 2024 şube sayısı (kaynaklarda bulunamadı)" in output.answer.answer


def test_figure_that_cannot_be_normalised_is_missing_with_that_reason(tmp_dir):
    # two numbers in one value: the shared post-processing refuses to guess
    sub = {QA: sub_reply("694"), QG: sub_reply("3 bin 2", "şube")}
    backend = Backend(plan=MAX_PLAN, sub=sub)
    pipeline, _ = make(tmp_dir, backend)
    output = run(pipeline)
    assert output.answer.abstained
    assert "Garanti BBVA 2024 şube sayısı (sayısal değer çıkarılamadı)" in output.answer.answer
    lookups = {lk["id"]: lk for lk in output.debug["router"]["lookups"]}
    assert lookups["g"]["normalization_error"]


def test_synthesis_facts_show_the_basis_only_for_money_and_ratios(tmp_dir):
    sub = {QA: sub_reply("694", basis="solo"), QG: sub_reply("794")}
    backend = Backend(plan=MAX_PLAN, sub=sub)
    pipeline, _ = make(tmp_dir, backend)
    run(pipeline)
    synth_user = next(u for s, _, u in backend.calls if s == "synthesize")
    assert "- Akbank 2024 şube sayısı: 694 (2024 raporu)\n" in synth_user
    assert "solo" not in synth_user


def test_unit_mismatch_is_an_abstention_with_the_reason(tmp_dir):
    sub = {QA: sub_reply("694"), QG: sub_reply("12,5", "%")}
    backend = Backend(plan=MAX_PLAN, sub=sub)
    pipeline, _ = make(tmp_dir, backend)
    output = run(pipeline)
    assert output.answer.abstained and "birim" in output.answer.answer
    assert "synthesize" not in steps(backend)


# --- final answer ----------------------------------------------------------------------


def test_value_and_unit_come_from_compute_not_from_the_synthesis_text(tmp_dir):
    q23, q24 = "Akbank 2023 şube?", "Akbank 2024 şube?"
    plan = {
        "expressible": True,
        "lookups": [lookup("a23", "akbank", 2023, q23, 2024), lookup("a24", "akbank", 2024, q24)],
        "operation": {"kind": "percent_change", "operands": ["a23", "a24"]},
    }
    sub = {q23: sub_reply("705"), q24: sub_reply("694")}
    backend = Backend(plan=plan, sub=sub, synth="Şube sayısı %50 arttı.")  # wrong on purpose
    pipeline, _ = make(tmp_dir, backend)
    output = run(pipeline)
    assert output.answer.answer == "Şube sayısı %50 arttı."
    assert (output.answer.value, output.answer.unit) == (-1.56, "%")
    synth_user = next(u for s, _, u in backend.calls if s == "synthesize")
    assert "%1,56 azalış" in synth_user  # the model is handed the computed result


def test_citations_are_the_union_of_the_lookups(tmp_dir):
    backend = Backend(plan=MAX_PLAN, sub=MAX_SUB)
    pipeline, _ = make(tmp_dir, backend)
    output = run(pipeline)
    assert set(map(tuple, (c.model_dump().values() for c in output.answer.citations))) == {
        ("akbank-2024", 11),
        ("akbank-2024", 12),
        ("garanti-2024", 11),
    }
    assert output.answer.citations[:2] == [
        Citation(doc="akbank-2024", page=11),
        Citation(doc="akbank-2024", page=12),
    ]


def test_retrieved_list_interleaves_the_lookups(tmp_dir):
    backend = Backend(plan=MAX_PLAN, sub=MAX_SUB)
    pipeline, _ = make(tmp_dir, backend, retriever=FakeRetriever(shared=True))
    output = run(pipeline)
    docs = [i.doc for i in output.retrieved]
    assert len(output.retrieved) == 10
    assert docs[0] == "shared-2024"  # first chunk of both lists, taken once
    assert docs[1:5] == ["akbank-2024", "garanti-2024", "akbank-2024", "garanti-2024"]
    assert len({i.chunk_id for i in output.retrieved}) == 10
    lookups = output.debug["router"]["lookups"]
    assert all(len(lk["retrieved"]) == 10 for lk in lookups)


def test_interleave_round_robin_dedup_and_cap():
    a = [item("a", n, 1) for n in range(3)]
    b = [a[0], item("b", 1, 2)]
    merged = interleave([a, b], 4)
    assert [i.chunk_id for i in merged] == [
        a[0].chunk_id,
        a[1].chunk_id,
        b[1].chunk_id,
        a[2].chunk_id,
    ]
    assert interleave([], 5) == []


def test_synthesis_failure_gives_a_templated_sentence(tmp_dir):
    backend = Backend(plan=MAX_PLAN, sub=MAX_SUB, synth=BackendError("down"))
    pipeline, _ = make(tmp_dir, backend)
    output = run(pipeline)
    answer = output.answer
    assert not answer.abstained
    assert (answer.value, answer.unit) == (794, "adet")
    assert "en yüksek" in answer.answer and "Garanti BBVA 2024 şube sayısı" in answer.answer
    assert "Akbank 2024 şube sayısı: 694" in answer.answer
    assert output.debug["router"]["synthesize"]["templated"] is True


# --- tags, debug, config ------------------------------------------------------------------


def test_every_llm_call_carries_the_runner_tags_and_its_step(tmp_dir):
    backend = Backend(plan=MAX_PLAN, sub=MAX_SUB)
    pipeline, _ = make(tmp_dir, backend)
    run(pipeline)
    rows = trace_rows(tmp_dir)
    assert len(rows) == 5
    for row in rows:
        assert {k: row["tags"][k] for k in TAGS} == TAGS
    by_step = sorted((r["tags"]["step"], r["model"], r["tags"].get("lookup")) for r in rows)
    assert by_step == [
        ("classify", "claude-haiku-5-5", None),
        ("decompose", "claude-sonnet-5-5", None),
        ("sub_answer", "claude-haiku-5-5", "a"),
        ("sub_answer", "claude-haiku-5-5", "g"),
        ("synthesize", "claude-haiku-5-5", None),
    ]


def test_debug_records_plan_lookups_and_computation(tmp_dir):
    backend = Backend(plan=MAX_PLAN, sub=MAX_SUB)
    pipeline, _ = make(tmp_dir, backend)
    output = run(pipeline)
    json.dumps(output.debug)  # serialisable for traces.jsonl
    router = output.debug["router"]
    assert router["classify"]["route"] == "multi"
    assert router["decompose"]["plan"]["operation"]["kind"] == "max"
    lk = router["lookups"][0]
    assert lk["id"] == "a" and lk["filter"] == {"banks": ["akbank"], "years": [2024]}
    assert lk["raw_value"] == "694" and lk["answer"]["value"] == 694
    assert router["compute"]["inputs"][1] == {
        "id": "g",
        "label": "Garanti BBVA 2024 şube sayısı",
        "value": 794,
        "unit": "adet",
        "basis": None,
        "missing": None,
    }


def test_access_role_applies_to_every_lookup(tmp_dir):
    backend = Backend(plan=MAX_PLAN, sub=MAX_SUB)
    params = {"rag": {**RAG, "access": {"role": "public"}}}
    pipeline, retriever = make(tmp_dir, backend, params=params)
    output = run(pipeline)
    for call in retriever.calls:
        values = list(call["where"].params.values())
        assert values[:2] in ([["akbank"], [2024]], [["garanti"], [2024]])
        assert len(values) == 3 and "public" in values[2]  # bank, year, access roles
    assert output.debug["access"]["role"] == "public"
    assert output.debug["access"]["source"] == "config"


def test_unknown_caller_role_is_refused_before_any_work(tmp_dir):
    backend = Backend()
    pipeline, retriever = make(tmp_dir, backend)
    with pytest.raises(ValueError):
        run(pipeline, access_role="nobody")
    assert backend.calls == [] and retriever.calls == []


def test_unexpected_exceptions_propagate(tmp_dir):
    class Broken(FakeRetriever):
        def retrieve(self, question, k, where=None):
            raise RuntimeError("db down")

    backend = Backend(plan=MAX_PLAN, sub=MAX_SUB)
    pipeline, _ = make(tmp_dir, backend, retriever=Broken())
    with pytest.raises(RuntimeError, match="db down"):
        run(pipeline)


def test_aclose_closes_the_retriever(tmp_dir):
    pipeline, retriever = make(tmp_dir, Backend())
    asyncio.run(pipeline.aclose())
    assert retriever.closed


def test_router_config_simple_path_is_the_rerank_config():
    from kaprag.agent.pipeline import build

    rerank, router = load_config("rerank"), load_config("router")
    assert router.topic == "router"
    assert resolve_factory(router.pipeline.factory) is build
    assert router.pipeline.params["rag"] == rerank.pipeline.params
    config = parse_router_config(router.pipeline.params)
    assert (config.classify.model, config.synthesize.model) == ("claude-haiku-5-5",) * 2
    assert (config.decompose.model, config.decompose.max_lookups) == ("claude-sonnet-5-5", 8)
    assert config.rag.access is None
    assert router.model_dump(exclude={"pipeline", "description", "topic"}) == rerank.model_dump(
        exclude={"pipeline", "description", "topic"}
    )


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"rag": RAG, "decompose": {"model": "claude-sonnet-5-5", "max_lookups": 9}},
        {"rag": RAG, "decompose": {"model": "claude-sonnet-5-5", "max_lookups": 1}},
        {"rag": RAG, "planner": {}},
        {"rag": {**RAG, "top_k": 3}},
    ],
)
def test_invalid_router_params_fail_loudly(params):
    from kaprag.agent.pipeline import RouterConfigError

    with pytest.raises(RouterConfigError):
        parse_router_config(params)
