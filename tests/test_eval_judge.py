"""LLM judge (with a fake backend), fake judge, and one optional real judge call."""

import asyncio
import json
from pathlib import Path

import pytest

from kaprag.eval.answer import SystemAnswer
from kaprag.eval.judge import (
    JUDGE_MODEL,
    KEY_POINTS_SYSTEM,
    TEXT_SYSTEM,
    FakeJudge,
    JudgeError,
    KeyPointVerdict,
    LLMJudge,
    key_point_coverage,
)
from kaprag.eval.schema import load_questions
from kaprag.llm import LLMClient, Usage, create_client
from kaprag.llm.types import BackendResponse

FIXTURE = Path(__file__).parent / "fixtures" / "eval" / "questions.jsonl"
Q = {q.id: q for q in load_questions(FIXTURE)}


class ReplyBackend:
    name = "fake"

    def __init__(self, reply):
        self.reply = reply
        self.requests = []

    async def generate(self, system, user, model):
        self.requests.append((system, user, model))
        return BackendResponse(text=self.reply, usage=Usage(input_tokens=50, output_tokens=5))


def client_for(backend, tmp_dir):
    return LLMClient(backend, cache_dir=None, trace_path=tmp_dir / "trace.jsonl")


def test_llm_judge_text_verdict_and_tags(tmp_dir):
    backend = ReplyBackend('{"reason": "same fact", "correct": true}')
    judge = LLMJudge(client_for(backend, tmp_dir))
    answer = SystemAnswer(answer="Garanti BBVA en yüksek.", value=None)
    verdict = asyncio.run(judge.judge_text(Q["f005"], answer, tags={"qid": "f005", "run": "r"}))

    assert verdict.correct is True
    system, user, model = backend.requests[0]
    assert (system, model) == (TEXT_SYSTEM, JUDGE_MODEL)
    assert "REFERENCE ANSWER:\nGaranti BBVA" in user
    assert "SYSTEM ANSWER:\nGaranti BBVA en yüksek." in user
    trace = json.loads((tmp_dir / "trace.jsonl").read_text(encoding="utf-8"))
    assert trace["tags"] == {"qid": "f005", "run": "r", "role": "judge", "judge": "text"}
    assert trace["model"] == "claude-sonnet-5-5"


def test_llm_judge_key_points(tmp_dir):
    backend = ReplyBackend('{"reason": "one of two", "covered": [true, false]}')
    judge = LLMJudge(client_for(backend, tmp_dir))
    verdict = asyncio.run(
        judge.judge_key_points(Q["f006"], SystemAnswer(answer="..."), tags={"qid": "f006"})
    )
    assert key_point_coverage(Q["f006"], verdict) == 0.5
    system, user, _ = backend.requests[0]
    assert system == KEY_POINTS_SYSTEM
    assert "KEY POINTS (2):\n1. yenilenebilir enerji finansmanı\n2. 2050 net sıfır hedefi" in user


def test_key_point_count_mismatch_raises():
    with pytest.raises(JudgeError, match="3 verdicts for 2"):
        key_point_coverage(Q["f006"], KeyPointVerdict(reason="", covered=[True] * 3))


def test_answer_block_includes_structured_value(tmp_dir):
    backend = ReplyBackend('{"reason": "x", "correct": false}')
    judge = LLMJudge(client_for(backend, tmp_dir))
    answer = SystemAnswer(answer="700", value=700, unit="adet", basis="konsolide")
    asyncio.run(judge.judge_text(Q["f001"], answer, tags={}))
    user = backend.requests[0][1]
    assert "(structured value: 700 adet)" in user
    assert "(basis: konsolide)" in user


def test_fake_judge_substring_match():
    judge = FakeJudge()

    def text_ok(answer):
        verdict = asyncio.run(judge.judge_text(Q["f005"], SystemAnswer(answer=answer), tags={}))
        return verdict.correct

    assert text_ok("Bence  garanti bbva.") is True
    assert text_ok("Akbank") is False
    assert text_ok("") is False
    kp = asyncio.run(
        judge.judge_key_points(
            Q["f006"], SystemAnswer(answer="Yenilenebilir enerji finansmanı artacak."), tags={}
        )
    )
    assert kp.covered == [True, False]


@pytest.mark.llm
def test_real_judge_call(tmp_dir):
    """The only real Claude call in the eval tests: one text verdict from the judge model."""
    client = create_client(trace_path=tmp_dir / "trace.jsonl")
    answer = SystemAnswer(answer="Sermaye yeterlilik oranı en yüksek banka Garanti BBVA'dır.")
    verdict = asyncio.run(LLMJudge(client).judge_text(Q["f005"], answer, tags={"qid": "f005"}))
    assert verdict.correct is True
