"""Metadata filter from the parsed question: schema, filter values, SQL, the parse call
(scripted LLM backend, fake retrievers; no GPU, DB or network)."""

import asyncio
import json
import threading

import pytest
from pydantic import ValidationError

from kaprag.corpus import BANK_NAMES, read_manifest
from kaprag.eval.pipeline import RetrievedItem
from kaprag.llm import BackendError, LLMClient, Usage
from kaprag.llm.types import BackendResponse
from kaprag.rag import retriever as retriever_module
from kaprag.rag.config import RagConfigError, parse_rag_config
from kaprag.rag.pipeline import RagPipeline
from kaprag.rag.query_filter import (
    CORPUS_YEARS,
    SYSTEM_PROMPT,
    ParsedQuery,
    filter_values,
    metadata_filter,
    parse_query_filter,
)
from kaprag.rag.rerank import RerankingRetriever
from kaprag.rag.retriever import DenseRetriever, HybridRetriever, RankedChunk
from kaprag.retrieval import SqlFilter

TAGS = {"run": "r1", "qid": "q007", "role": "pipeline"}
ANSWER = json.dumps({"answer": "Kaynaklarda yok.", "abstained": True, "citations": []})
PARAMS = {"index": "structure", "query_filter": {"enabled": True, "model": "claude-haiku-5-5"}}


def parsed(banks=(), years=()):
    return ParsedQuery(banks=list(banks), years=list(years))


def render(where: SqlFilter) -> str:
    return where.condition.as_string(None)


# --- schema and filter values -------------------------------------------------------


def test_schema_bank_keys_are_the_corpus_banks():
    assert ParsedQuery.model_json_schema()["properties"]["banks"]["items"]["enum"] == list(
        BANK_NAMES
    )


def test_corpus_years_match_the_manifest():
    assert tuple(sorted({d.year for d in read_manifest()})) == CORPUS_YEARS


@pytest.mark.parametrize(
    "payload",
    [
        {"banks": ["ziraat"], "years": []},
        {"banks": ["Akbank"], "years": []},
        {"banks": [], "years": ["2024"]},
        {"banks": [], "years": [True]},
        {"banks": [], "years": [2024.5]},
    ],
)
def test_schema_rejects_unknown_banks_and_non_integer_years(payload):
    with pytest.raises(ValidationError):
        ParsedQuery.model_validate_json(json.dumps(payload))


def test_missing_fields_default_to_empty():
    assert ParsedQuery.model_validate_json("{}") == parsed()


@pytest.mark.parametrize(
    ("banks", "years", "expected"),
    [
        (["akbank"], [], (["akbank"], [])),  # banks only
        ([], [2024], ([], [2024])),  # years only
        (
            ["yapikredi", "garanti", "garanti"],
            [2025, 2024],
            (["garanti", "yapikredi"], [2024, 2025]),
        ),
        ([], [], ([], [])),  # none
        (["isbank"], [2022], (["isbank"], [])),  # out-of-corpus year dropped
        (["isbank"], [2022, 2023], (["isbank"], [2023])),  # ... the others are kept
    ],
)
def test_filter_values(banks, years, expected):
    assert filter_values(parsed(banks, years)) == expected


def test_no_values_means_no_filter():
    assert metadata_filter([], []) is None


def test_banks_only_filter():
    where = metadata_filter(["akbank", "garanti"], [])
    assert render(where) == '"bank" = ANY(%(f0)s)'
    assert where.params == {"f0": ["akbank", "garanti"]}


def test_years_only_filter():
    where = metadata_filter([], [2024])
    assert render(where) == '"year" = ANY(%(f0)s)'
    assert where.params == {"f0": [2024]}


def test_bank_and_year_filter_is_one_and_with_bound_parameters():
    where = metadata_filter(["isbank"], [2023, 2025])
    assert render(where) == '"bank" = ANY(%(f0)s) AND "year" = ANY(%(f1)s)'
    assert where.params == {"f0": ["isbank"], "f1": [2023, 2025]}


@pytest.mark.parametrize(
    ("banks", "years"),
    [
        (["akbank'; DROP TABLE chunks_fixed; --"], []),
        (["ziraat"], []),
        ([], ["2024"]),
        ([], [True]),
    ],
)
def test_filter_builder_rejects_unvalidated_values(banks, years):
    with pytest.raises(ValueError):
        metadata_filter(banks, years)


# --- the composable filter ------------------------------------------------------------


def test_sql_filter_composes_conditions_with_distinct_parameters():
    where = SqlFilter().any_of("bank", ["akbank"]).any_of("doc_id", ["akbank-2024", "x'y"])
    assert render(where) == '"bank" = ANY(%(f0)s) AND "doc_id" = ANY(%(f1)s)'
    assert where.params == {"f0": ["akbank"], "f1": ["akbank-2024", "x'y"]}
    assert "x'y" not in render(where)  # values never enter the SQL text


def test_sql_filter_is_immutable_and_empty_by_default():
    empty = SqlFilter()
    assert not empty and empty.params == {}
    one = empty.any_of("bank", ["akbank"])
    assert one and not empty


def test_sql_filter_refuses_an_empty_value_list():
    with pytest.raises(ValueError, match="no values"):
        SqlFilter().any_of("bank", [])


# --- the parse call ---------------------------------------------------------------------


class ScriptedBackend:
    name = "fake"

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    async def generate(self, system, user, model):
        self.prompts.append((system, user, model))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return BackendResponse(text=reply, usage=Usage(200, 20))


def client_for(tmp_dir, replies):
    backend = ScriptedBackend(replies)
    return LLMClient(backend, cache_dir=None, trace_path=tmp_dir / "trace.jsonl"), backend


def trace_rows(tmp_dir):
    text = (tmp_dir / "trace.jsonl").read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines()]


def test_parse_call_is_a_tagged_pipeline_call_with_its_own_step(tmp_dir):
    client, backend = client_for(tmp_dir, ['{"banks": ["akbank"], "years": [2024]}'])
    outcome = asyncio.run(
        parse_query_filter(client, "Akbank 2024 net kâr?", model="claude-haiku-5-5", tags=TAGS)
    )
    system, user, model = backend.prompts[0]
    assert system == SYSTEM_PROMPT and model == "claude-haiku-5-5"
    assert user.startswith("Soru: Akbank 2024 net kâr?")
    (row,) = trace_rows(tmp_dir)
    assert row["tags"] == {**TAGS, "step": "parse_query"}
    assert render(outcome.where) == '"bank" = ANY(%(f0)s) AND "year" = ANY(%(f1)s)'
    assert outcome.debug == {
        "parsed": {"banks": ["akbank"], "years": [2024]},
        "applied": {"banks": ["akbank"], "years": [2024]},
        "error": None,
    }


def test_nothing_usable_parsed_means_no_filter(tmp_dir):
    client, _ = client_for(tmp_dir, ['{"banks": [], "years": [2022]}'])
    outcome = asyncio.run(parse_query_filter(client, "soru", model="claude-haiku-5-5", tags=TAGS))
    assert outcome.where is None
    assert outcome.debug == {
        "parsed": {"banks": [], "years": [2022]},
        "applied": None,
        "error": None,
    }


@pytest.mark.parametrize(
    "replies",
    [
        ["not json", '{"banks": ["ziraat"]}'],  # invalid twice (the client retries once)
        [BackendError("backend down")],  # a non-retryable backend error
    ],
)
def test_parse_failure_means_no_filter_and_is_recorded(tmp_dir, replies):
    client, _ = client_for(tmp_dir, replies)
    outcome = asyncio.run(parse_query_filter(client, "soru", model="claude-haiku-5-5", tags=TAGS))
    assert outcome.where is None
    assert outcome.debug["parsed"] is None and outcome.debug["applied"] is None
    assert outcome.debug["error"]


# --- the pipeline -----------------------------------------------------------------------


def items(n):
    return [
        RetrievedItem(
            chunk_id=f"akbank-2024:structure:{i}",
            doc="akbank-2024",
            page_start=i,
            page_end=i,
            score=1.0 - i / 100,
            text=f"chunk {i}",
        )
        for i in range(1, n + 1)
    ]


class FakeRetriever:
    def __init__(self, n=12):
        self.n = n
        self.calls = []
        self.thread = None

    def retrieve(self, question, k, where=None):
        self.calls.append((question, k, where))
        self.thread = threading.current_thread()
        return [RankedChunk(i) for i in items(self.n)][:k]

    def close(self):
        pass


def make_pipeline(tmp_dir, replies, params=PARAMS, n=12):
    client, backend = client_for(tmp_dir, replies)
    retriever = FakeRetriever(n)
    return RagPipeline(parse_rag_config(params), retriever, lambda: client), backend, retriever


def test_pipeline_parses_then_retrieves_with_the_filter(tmp_dir):
    pipeline, backend, retriever = make_pipeline(
        tmp_dir, ['{"banks": ["akbank"], "years": [2024]}', ANSWER]
    )
    output = asyncio.run(pipeline.run("Akbank 2024 net kâr?", tags=dict(TAGS)))

    ((question, k, where),) = retriever.calls
    assert (question, k) == ("Akbank 2024 net kâr?", 10)
    assert where.params == {"f0": ["akbank"], "f1": [2024]}
    assert [row["tags"]["step"] for row in trace_rows(tmp_dir)] == ["parse_query", "generate"]
    assert output.debug["query_filter"] == {
        "parsed": {"banks": ["akbank"], "years": [2024]},
        "applied": {"banks": ["akbank"], "years": [2024]},
        "error": None,
        "retrieved": 10,
    }
    json.dumps(output.debug)


def test_pipeline_parse_failure_retrieves_unfiltered_and_answers(tmp_dir):
    pipeline, backend, retriever = make_pipeline(tmp_dir, ["not json", "still not json", ANSWER])
    output = asyncio.run(pipeline.run("soru", tags=dict(TAGS)))

    assert retriever.calls == [("soru", 10, None)]
    assert output.answer.abstained  # the question proceeds to generation
    qf = output.debug["query_filter"]
    assert qf["parsed"] is None and qf["applied"] is None and qf["error"]
    assert qf["retrieved"] == 10


def test_filter_leaving_few_candidates_is_not_widened(tmp_dir):
    pipeline, backend, retriever = make_pipeline(
        tmp_dir, ['{"banks": ["akbank"], "years": []}', ANSWER], n=2
    )
    output = asyncio.run(pipeline.run("soru", tags=dict(TAGS)))
    assert len(retriever.calls) == 1  # no second, unfiltered retrieval
    assert len(output.retrieved) == 2
    assert output.debug["query_filter"]["retrieved"] == 2


def test_pipeline_with_an_empty_filtered_result_generates_from_an_empty_context(tmp_dir):
    pipeline, backend, _ = make_pipeline(tmp_dir, ['{"banks": ["akbank"]}', ANSWER], n=0)
    output = asyncio.run(pipeline.run("soru", tags=dict(TAGS)))
    assert output.retrieved == [] and output.answer.abstained
    assert backend.prompts[1][1].startswith("Soru: soru\n\nKaynaklar:\n\n\n\nRespond with")


@pytest.mark.parametrize(
    "params", [{"index": "structure"}, {**PARAMS, "query_filter": {"enabled": False}}]
)
def test_without_the_query_filter_there_is_no_parse_call(tmp_dir, params):
    pipeline, backend, retriever = make_pipeline(tmp_dir, [ANSWER], params=params)
    output = asyncio.run(pipeline.run("Akbank 2024?", tags=dict(TAGS)))
    assert retriever.calls == [("Akbank 2024?", 10, None)]
    assert len(backend.prompts) == 1
    assert "query_filter" not in output.debug


# --- the filter reaches every retriever -------------------------------------------------


class FakeEmbedder:
    def embed_query(self, text):
        return [0.1, 0.2]


@pytest.fixture
def arms(monkeypatch):
    calls = {"dense": [], "fts": []}

    def dense_search(conn, variant, vector, k=10, where=None):
        calls["dense"].append(where)
        return []

    def fts_search(conn, variant, query, k=10, where=None):
        calls["fts"].append(where)
        return []

    monkeypatch.setattr(retriever_module, "dense_search", dense_search)
    monkeypatch.setattr(retriever_module, "fts_search", fts_search)
    return calls


def test_filter_reaches_both_arms_of_the_hybrid_retriever(arms):
    where = metadata_filter(["akbank"], [2024])
    retriever = HybridRetriever(None, "structure", FakeEmbedder(), candidates_per_arm=40, rrf_k=60)
    retriever.retrieve("soru", 10, where=where)
    retriever.retrieve("soru", 10)
    assert arms["dense"] == [where, None] and arms["fts"] == [where, None]


def test_filter_reaches_the_dense_retriever(arms):
    where = metadata_filter([], [2025])
    retriever = DenseRetriever(None, "structure", FakeEmbedder())
    retriever.retrieve("soru", 10, where=where)
    retriever.retrieve("soru", 10)
    assert arms["dense"] == [where, None] and arms["fts"] == []


def test_reranking_retriever_passes_the_filter_to_the_first_stage():
    class Scorer:
        def score(self, question, passages):
            return [0.0] * len(passages)

    inner = FakeRetriever(n=3)
    where = metadata_filter(["garanti"], [])
    RerankingRetriever(inner, Scorer(), 40).retrieve("soru", 10, where=where)
    assert inner.calls == [("soru", 40, where)]


# --- config --------------------------------------------------------------------------------


def test_query_filter_config_defaults_and_validation():
    assert parse_rag_config({"index": "fixed"}).query_filter is None
    qf = parse_rag_config({"index": "fixed", "query_filter": {}}).query_filter
    assert (qf.enabled, qf.model) == (True, "claude-haiku-5-5")
    with pytest.raises(RagConfigError, match="banks"):
        parse_rag_config({"index": "fixed", "query_filter": {"banks": ["akbank"]}})
    with pytest.raises(RagConfigError, match="enabled"):
        parse_rag_config({"index": "fixed", "query_filter": {"enabled": "maybe"}})
