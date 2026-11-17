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
        ({"index": "fixed", "retrieval": {"mode": "hybrid"}}, "unknown retrieval mode 'hybrid'"),
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
