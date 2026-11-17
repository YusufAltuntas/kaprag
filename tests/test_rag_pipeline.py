"""RagPipeline with a fake retriever and a scripted LLM backend (no GPU, DB or network)."""

import asyncio
import json
import threading

import pytest

from kaprag.eval.answer import Citation
from kaprag.eval.pipeline import RetrievedItem
from kaprag.llm import LLMCallError, LLMClient, Usage
from kaprag.llm.types import BackendResponse
from kaprag.rag.config import parse_rag_config
from kaprag.rag.generate import SYSTEM_PROMPT
from kaprag.rag.pipeline import RagPipeline
from kaprag.rag.retriever import RankedChunk

TAGS = {"run": "r1", "qid": "q007", "role": "pipeline"}


def items(n):
    return [
        RetrievedItem(
            chunk_id=f"akbank-2024:fixed:{i}",
            doc="akbank-2024",
            page_start=10 + i,
            page_end=10 + i,
            score=1.0 - i / 100,
            text=f"chunk {i} metni",
        )
        for i in range(1, n + 1)
    ]


class FakeRetriever:
    def __init__(self, n=12, traces=None):
        self.items = items(n)
        self.traces = traces or [{} for _ in self.items]
        self.calls = []
        self.closed = False
        self.thread = None

    def retrieve(self, question, k):
        self.calls.append((question, k))
        self.thread = threading.current_thread()
        ranked = [RankedChunk(i, t) for i, t in zip(self.items, self.traces, strict=True)]
        return ranked[:k]

    def close(self):
        self.closed = True


class ScriptedBackend:
    name = "fake"

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    async def generate(self, system, user, model):
        self.prompts.append((system, user, model))
        return BackendResponse(text=self.replies.pop(0), usage=Usage(1000, 100))


def make(tmp_dir, replies, params=None, retriever=None):
    backend = ScriptedBackend(replies)
    client = LLMClient(backend, cache_dir=None, trace_path=tmp_dir / "trace.jsonl")
    config = parse_rag_config(params or {"index": "fixed", "retrieve_k": 10, "context_k": 5})
    retriever = retriever or FakeRetriever()
    return RagPipeline(config, retriever, lambda: client), backend, retriever


def trace_rows(tmp_dir):
    text = (tmp_dir / "trace.jsonl").read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines()]


REPLY = json.dumps(
    {
        "answer": "Net kâr 42,1 milyar TL [2].",
        "value": "42,1",
        "unit": "milyar TL",
        "basis": "konsolide",
        "citations": [2, 9],
        "abstained": False,
        "clarification": None,
    }
)


def test_returns_retrieve_k_ranked_items_and_generates_from_context_k(tmp_dir):
    pipeline, backend, retriever = make(tmp_dir, [REPLY])
    output = asyncio.run(pipeline.run("Akbank 2024 net kâr?", tags=dict(TAGS)))

    assert retriever.calls == [("Akbank 2024 net kâr?", 10)]
    assert retriever.thread is not threading.main_thread()  # blocking work off the loop
    assert [i.chunk_id for i in output.retrieved] == [i.chunk_id for i in items(10)]

    system, user, model = backend.prompts[0]
    assert system == SYSTEM_PROMPT
    assert model == "claude-haiku-5-5"
    assert user.startswith("Soru: Akbank 2024 net kâr?\n\nKaynaklar:\n\n[1] Banka: Akbank")
    assert "[5] Banka" in user and "chunk 5 metni" in user
    assert "[6] Banka" not in user and "chunk 6 metni" not in user

    answer = output.answer
    assert (answer.value, answer.unit, answer.basis) == (42_100_000_000, "TL", "konsolide")
    assert answer.citations == [Citation(doc="akbank-2024", page=12)]  # source 9 dropped
    assert output.debug["dropped_citations"] == [9]
    assert output.debug["context_chunk_ids"] == [i.chunk_id for i in items(5)]
    assert "retrieval_trace" not in output.debug  # dense: no per-stage ranks to record


def test_llm_call_carries_the_runner_tags(tmp_dir):
    pipeline, _, _ = make(tmp_dir, [REPLY])
    asyncio.run(pipeline.run("soru", tags=dict(TAGS)))
    (row,) = trace_rows(tmp_dir)
    assert row["tags"] == {**TAGS, "step": "generate"}
    assert row["model"] == "claude-haiku-5-5"


def test_invalid_json_twice_propagates_as_an_exception(tmp_dir):
    pipeline, backend, _ = make(tmp_dir, ["not json", '{"answer": 3}'])
    with pytest.raises(LLMCallError):
        asyncio.run(pipeline.run("soru", tags=dict(TAGS)))
    assert len(backend.prompts) == 2
    (row,) = trace_rows(tmp_dir)
    assert row["error"] and row["tags"]["qid"] == "q007"


def test_fewer_hits_than_k(tmp_dir):
    abstain = json.dumps({"answer": "Kaynaklarda yok.", "abstained": True, "citations": []})
    pipeline, backend, _ = make(tmp_dir, [abstain], retriever=FakeRetriever(n=3))
    output = asyncio.run(pipeline.run("soru", tags=dict(TAGS)))
    assert len(output.retrieved) == 3
    assert output.answer.abstained and output.answer.value is None
    assert "[3] Banka" in backend.prompts[0][1]


def test_aclose_closes_the_retriever(tmp_dir):
    pipeline, _, retriever = make(tmp_dir, [])
    asyncio.run(pipeline.aclose())
    assert retriever.closed


def test_numeric_value_is_rejected_then_retried_then_surfaced(tmp_dir):
    number = json.dumps({"answer": "2.202 milyar TL [1].", "value": 2.202, "unit": "milyar TL"})
    pipeline, backend, _ = make(tmp_dir, [number, number])
    with pytest.raises(LLMCallError, match="GeneratedAnswer"):
        asyncio.run(pipeline.run("soru", tags=dict(TAGS)))
    assert len(backend.prompts) == 2  # the client's single JSON retry


def test_prompt_states_citation_and_basis_rules():
    assert "reddettiğin" in SYSTEM_PROMPT  # cite only supporting sources, not rejected ones
    assert "varsaydıysan" in SYSTEM_PROMPT  # an assumed consolidated figure sets `basis`


def test_retrieval_trace_records_each_returned_item(tmp_dir):
    traces = [{"dense_rank": i, "fts_rank": None} for i in range(1, 13)]
    pipeline, _, _ = make(tmp_dir, [REPLY], retriever=FakeRetriever(traces=traces))
    output = asyncio.run(pipeline.run("soru", tags=dict(TAGS)))
    assert output.debug["retrieval_trace"] == [
        {"chunk_id": item.chunk_id, "dense_rank": i, "fts_rank": None}
        for i, item in enumerate(items(10), start=1)
    ]
