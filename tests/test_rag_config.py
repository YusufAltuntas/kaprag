"""RAG pipeline params and the shipped `baseline` config."""

import pytest

from kaprag.eval.config import load_config
from kaprag.eval.pipeline import resolve_factory
from kaprag.rag.config import RagConfigError, parse_rag_config
from kaprag.rag.pipeline import build


def test_baseline_config_values():
    cfg = load_config("baseline")
    assert cfg.topic == "baseline"
    assert cfg.judge.kind == "llm"
    assert resolve_factory(cfg.pipeline.factory) is build
    rag = parse_rag_config(cfg.pipeline.params)
    assert rag.index == "fixed"
    assert rag.retrieval.mode == "dense"
    assert (rag.retrieve_k, rag.context_k) == (10, 5)
    assert rag.generation.model == "claude-haiku-5-5"
    assert (rag.embedding.model, rag.embedding.device) == ("BAAI/bge-m3", "cuda")
    assert cfg.pipeline.params["retrieval"] == {"mode": "dense"}  # nothing new in the baseline


def test_hybrid_config_is_the_baseline_with_only_the_retrieval_block_changed():
    baseline, hybrid = load_config("baseline"), load_config("hybrid")
    assert hybrid.topic == "hybrid"
    assert hybrid.pipeline.params["retrieval"] == {
        "mode": "hybrid",
        "candidates_per_arm": 40,
        "rrf_k": 60,
    }
    assert {k: v for k, v in hybrid.pipeline.params.items() if k != "retrieval"} == {
        k: v for k, v in baseline.pipeline.params.items() if k != "retrieval"
    }
    assert hybrid.model_dump(exclude={"pipeline", "description", "topic"}) == baseline.model_dump(
        exclude={"pipeline", "description", "topic"}
    )
    rag = parse_rag_config(hybrid.pipeline.params)
    assert (rag.retrieval.mode, rag.retrieval.candidates_per_arm, rag.retrieval.rrf_k) == (
        "hybrid",
        40,
        60,
    )


def test_defaults_need_only_the_index():
    rag = parse_rag_config({"index": "structure"})
    assert (rag.index, rag.retrieval.mode, rag.retrieve_k, rag.context_k) == (
        "structure",
        "dense",
        10,
        5,
    )


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"index": "fixed", "retrieval": {"mode": "sparse"}}, "unknown retrieval mode 'sparse'"),
        (
            {"index": "fixed", "retrieval": {"mode": "dense", "candidates_per_arm": 40}},
            "only apply to retrieval mode 'hybrid'",
        ),
        ({"index": "fixed", "retrieval": {"mode": "hybrid", "fusion": "rrf"}}, "fusion"),
        (
            {"index": "fixed", "retrieval": {"mode": "hybrid", "candidates_per_arm": 0}},
            "candidates_per_arm",
        ),
        ({"index": "fixed", "retrieval": {"mode": "hybrid", "rrf_k": -1}}, "rrf_k"),
        ({"index": "chunks"}, "index"),
        ({}, "index"),
        ({"index": "fixed", "context_k": 11}, "context_k"),
        ({"index": "fixed", "retrieve_k": 0}, "retrieve_k"),
        ({"index": "fixed", "top_k": 5}, "top_k"),
        ({"index": "fixed", "retrieval": {"mode": "dense", "rerank": True}}, "rerank"),
    ],
)
def test_invalid_params_fail_loudly(params, message):
    with pytest.raises(RagConfigError, match=message):
        parse_rag_config(params)
