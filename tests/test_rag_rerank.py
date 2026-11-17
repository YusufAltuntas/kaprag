"""Cross-encoder reranking stage with a fake scorer (no GPU, DB or network)."""

import asyncio
import json

import pytest

from kaprag.eval.pipeline import RetrievedItem
from kaprag.llm import LLMClient, Usage
from kaprag.llm.types import BackendResponse
from kaprag.rag import rerank as rerank_module
from kaprag.rag import retriever as retriever_module
from kaprag.rag.config import RagConfigError, parse_rag_config
from kaprag.rag.pipeline import RagPipeline
from kaprag.rag.rerank import RerankingRetriever, rerank
from kaprag.rag.retriever import RankedChunk, open_retriever

TAGS = {"run": "r1", "qid": "q007", "role": "pipeline"}
REPLY = json.dumps({"answer": "Kaynaklarda yok.", "abstained": True, "citations": []})


class ScriptedBackend:
    name = "fake"

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    async def generate(self, system, user, model):
        self.prompts.append((system, user, model))
        return BackendResponse(text=self.replies.pop(0), usage=Usage(1000, 100))


def ranked(n, trace=None):
    return [
        RankedChunk(
            RetrievedItem(
                chunk_id=f"c{i}",
                doc="garanti-2024",
                page_start=i,
                page_end=i,
                score=float(100 - i),
                text=f"passage {i}",
            ),
            dict(trace or {}),
        )
        for i in range(1, n + 1)
    ]


class FakeScorer:
    """Scores a passage by a lookup on its text; records every call."""

    def __init__(self, scores=None):
        self.scores = scores or {}
        self.calls = []

    def score(self, question, passages):
        self.calls.append((question, list(passages)))
        return [self.scores.get(p, 0.0) for p in passages]


class FakeRetriever:
    def __init__(self, n=50, trace=None):
        self.items = ranked(n, trace)
        self.calls = []
        self.closed = False

    def retrieve(self, question, k, where=None):
        assert where is None
        self.calls.append((question, k))
        return self.items[:k]

    def close(self):
        self.closed = True


def test_orders_by_score_and_keeps_the_first_stage_rank():
    scorer = FakeScorer({"passage 3": 0.9, "passage 1": 0.5, "passage 2": 0.7})
    out = rerank("soru", ranked(3, {"dense_rank": 7}), scorer, candidates=40)

    assert scorer.calls == [("soru", ["passage 1", "passage 2", "passage 3"])]
    assert [r.item.chunk_id for r in out] == ["c3", "c2", "c1"]
    assert [r.item.score for r in out] == [0.9, 0.7, 0.5]
    assert out[0].trace == {"dense_rank": 7, "first_stage_rank": 3, "first_stage_score": 97.0}
    assert out[0].item.text == "passage 3"


def test_ties_keep_the_first_stage_order():
    out = rerank("soru", ranked(3), FakeScorer(), candidates=40)
    assert [r.item.chunk_id for r in out] == ["c1", "c2", "c3"]


def test_only_the_top_candidates_are_scored_and_returned():
    scorer = FakeScorer({"passage 5": 1.0})
    out = rerank("soru", ranked(10), scorer, candidates=4)
    assert scorer.calls[0][1] == ["passage 1", "passage 2", "passage 3", "passage 4"]
    assert [r.item.chunk_id for r in out] == ["c1", "c2", "c3", "c4"]


def test_empty_candidate_list_skips_the_scorer():
    scorer = FakeScorer()
    assert rerank("soru", [], scorer, candidates=40) == []
    assert scorer.calls == []


def test_a_chunk_without_text_fails_loudly():
    item = RetrievedItem(chunk_id="x", doc="d", page_start=1, page_end=1, score=1.0)
    with pytest.raises(ValueError, match="no text"):
        rerank("soru", [RankedChunk(item)], FakeScorer(), candidates=40)


def test_reranking_retriever_asks_the_first_stage_for_the_candidates():
    inner = FakeRetriever(n=50)
    scorer = FakeScorer({"passage 40": 2.0, "passage 41": 3.0})
    retriever = RerankingRetriever(inner, scorer, candidates=40)
    out = retriever.retrieve("soru", 10)
    assert inner.calls == [("soru", 40)]
    assert len(out) == 10
    assert out[0].item.chunk_id == "c40"  # passage 41 was never a candidate
    assert out[0].trace["first_stage_rank"] == 40
    retriever.close()
    assert inner.closed


def test_pipeline_returns_and_generates_from_the_reranked_order(tmp_dir):
    inner = FakeRetriever(n=50, trace={"dense_rank": None, "fts_rank": 1})
    scorer = FakeScorer({f"passage {i}": i / 100 for i in range(1, 51)})  # reverses top-40
    backend = ScriptedBackend([REPLY])
    client = LLMClient(backend, cache_dir=None, trace_path=tmp_dir / "trace.jsonl")
    config = parse_rag_config({"index": "fixed", "retrieval": {"mode": "hybrid"}, "rerank": {}})
    pipeline = RagPipeline(config, RerankingRetriever(inner, scorer, 40), lambda: client)

    output = asyncio.run(pipeline.run("soru", tags=dict(TAGS)))

    assert [i.chunk_id for i in output.retrieved] == [f"c{i}" for i in range(40, 30, -1)]
    assert output.debug["context_chunk_ids"] == [f"c{i}" for i in range(40, 35, -1)]
    user = backend.prompts[0][1]
    assert user.index("passage 40") < user.index("passage 36")
    assert "passage 35" not in user
    assert output.debug["retrieval_trace"][0] == {
        "chunk_id": "c40",
        "dense_rank": None,
        "fts_rank": 1,
        "first_stage_rank": 40,
        "first_stage_score": 60.0,
    }
    json.dumps(output.debug)  # serialisable for traces.jsonl


@pytest.mark.parametrize("mode", ["dense", "hybrid"])
def test_open_retriever_wraps_any_mode_when_rerank_is_configured(monkeypatch, mode):
    opened, scorers = [], []

    def fake_opener(config):
        opened.append(config.retrieval.mode)
        return FakeRetriever()

    class FakeCrossEncoderScorer(FakeScorer):
        def __init__(self, model_name, *, device, batch_size, max_length):
            super().__init__()
            scorers.append((model_name, device, batch_size, max_length))

    monkeypatch.setitem(retriever_module.RETRIEVAL_MODES, mode, fake_opener)
    monkeypatch.setattr(rerank_module, "CrossEncoderScorer", FakeCrossEncoderScorer)
    config = parse_rag_config(
        {"index": "structure", "retrieval": {"mode": mode}, "rerank": {"candidates": 30}}
    )
    retriever = open_retriever(config)

    assert isinstance(retriever, RerankingRetriever)
    assert (opened, retriever.candidates) == ([mode], 30)
    assert scorers == [("BAAI/bge-reranker-v2-m3", "cuda", 8, 1024)]


def test_open_retriever_without_rerank_returns_the_first_stage(monkeypatch):
    first = FakeRetriever()
    monkeypatch.setitem(retriever_module.RETRIEVAL_MODES, "dense", lambda config: first)
    assert open_retriever(parse_rag_config({"index": "fixed"})) is first


def test_open_retriever_closes_the_first_stage_if_the_reranker_fails(monkeypatch):
    first = FakeRetriever()

    def broken(*args, **kwargs):
        raise RuntimeError("CUDA is not available")

    monkeypatch.setitem(retriever_module.RETRIEVAL_MODES, "dense", lambda config: first)
    monkeypatch.setattr(rerank_module, "CrossEncoderScorer", broken)
    with pytest.raises(RuntimeError, match="CUDA"):
        open_retriever(parse_rag_config({"index": "fixed", "rerank": {}}))
    assert first.closed


def test_rerank_config_defaults_and_validation():
    config = parse_rag_config({"index": "fixed", "rerank": {}})
    assert config.rerank.model_dump() == {
        "model": "BAAI/bge-reranker-v2-m3",
        "device": "cuda",
        "candidates": 40,
        "batch_size": 8,
        "max_length": 1024,
    }
    assert parse_rag_config({"index": "fixed"}).rerank is None
    for params, message in [
        ({"index": "fixed", "rerank": {"top_n": 5}}, "top_n"),
        ({"index": "fixed", "rerank": {"candidates": 9}}, "candidates"),
        ({"index": "fixed", "rerank": {"batch_size": 0}}, "batch_size"),
        ({"index": "fixed", "rerank": True}, "rerank"),
    ]:
        with pytest.raises(RagConfigError, match=message):
            parse_rag_config(params)
