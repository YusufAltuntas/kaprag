"""LLM judge for text correctness and key-point coverage (eval/README.md §6).

`LLMJudge` calls `claude-sonnet-5-5` through `kaprag.llm` with a structured verdict.
`FakeJudge` is a deterministic substring matcher for harness tests and fake runs only;
its numbers mean nothing about answer quality.

Every judge call is tagged `role: "judge"` (plus the caller's tags, e.g. `qid`), so the
runner can keep judge cost out of the pipeline's cost and latency.
"""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel

from kaprag.eval.answer import SystemAnswer
from kaprag.eval.schema import Question
from kaprag.llm import LLMClient

JUDGE_MODEL = "claude-sonnet-5-5"


class TextVerdict(BaseModel):
    reason: str
    correct: bool


class KeyPointVerdict(BaseModel):
    reason: str
    covered: list[bool]


class JudgeError(Exception):
    """The judge returned something unusable (e.g. the wrong number of verdicts)."""


class Judge(Protocol):
    kind: str

    async def judge_text(
        self, question: Question, answer: SystemAnswer, *, tags: dict[str, Any]
    ) -> TextVerdict: ...

    async def judge_key_points(
        self, question: Question, answer: SystemAnswer, *, tags: dict[str, Any]
    ) -> KeyPointVerdict: ...


def key_point_coverage(question: Question, verdict: KeyPointVerdict) -> float:
    """Fraction of the question's key points the verdict marks as covered."""
    n = len(question.key_points)
    if len(verdict.covered) != n:
        raise JudgeError(f"judge returned {len(verdict.covered)} verdicts for {n} key points")
    return sum(verdict.covered) / n


TEXT_SYSTEM = """\
You grade answers of a question-answering system over the annual reports of Turkish \
banks. Questions and answers are in Turkish. Compare the SYSTEM ANSWER with the \
REFERENCE ANSWER and decide whether the system answer is correct.

Correct: it states the same essential fact(s) as the reference (same entity, year and \
figure; numbers may be rounded or written in another unit, e.g. "45,2 milyar TL" vs \
"45.213 milyon TL"). Extra details that do not contradict the reference are fine.
Wrong: it contradicts the reference, misses the essential fact, gives several \
candidate answers without committing, or refuses to answer.
Judge content only, not style or language. Give a one-sentence reason in English."""

KEY_POINTS_SYSTEM = """\
You grade answers of a question-answering system over the annual reports of Turkish \
banks. Questions and answers are in Turkish. For each numbered KEY POINT decide \
whether the SYSTEM ANSWER covers it. A paraphrase or a more specific statement counts \
as covered; a vague mention that drops the substance does not. Return `covered` with \
exactly one boolean per key point, in order. Give a one-sentence reason in English."""


def _answer_block(answer: SystemAnswer) -> str:
    lines = [answer.answer.strip() or "(empty)"]
    if answer.value is not None:
        lines.append(f"(structured value: {answer.value} {answer.unit or ''})".rstrip())
    if answer.basis is not None:
        lines.append(f"(basis: {answer.basis})")
    return "\n".join(lines)


def text_prompt(question: Question, answer: SystemAnswer) -> str:
    return (
        f"QUESTION:\n{question.question}\n\n"
        f"REFERENCE ANSWER:\n{question.expected.answer}\n\n"
        f"SYSTEM ANSWER:\n{_answer_block(answer)}"
    )


def key_points_prompt(question: Question, answer: SystemAnswer) -> str:
    points = "\n".join(f"{i}. {kp}" for i, kp in enumerate(question.key_points, start=1))
    return (
        f"QUESTION:\n{question.question}\n\n"
        f"KEY POINTS ({len(question.key_points)}):\n{points}\n\n"
        f"SYSTEM ANSWER:\n{_answer_block(answer)}"
    )


class LLMJudge:
    kind = "llm"

    def __init__(self, client: LLMClient, model: str = JUDGE_MODEL) -> None:
        self.client = client
        self.model = model

    async def judge_text(
        self, question: Question, answer: SystemAnswer, *, tags: dict[str, Any]
    ) -> TextVerdict:
        result = await self.client.complete(
            TEXT_SYSTEM,
            text_prompt(question, answer),
            self.model,
            schema=TextVerdict,
            tags={**tags, "role": "judge", "judge": "text"},
        )
        return result.parsed

    async def judge_key_points(
        self, question: Question, answer: SystemAnswer, *, tags: dict[str, Any]
    ) -> KeyPointVerdict:
        result = await self.client.complete(
            KEY_POINTS_SYSTEM,
            key_points_prompt(question, answer),
            self.model,
            schema=KeyPointVerdict,
            tags={**tags, "role": "judge", "judge": "key_points"},
        )
        return result.parsed


def _fold(text: str) -> str:
    return " ".join(text.replace("İ", "i").replace("I", "ı").lower().split())


class FakeJudge:
    """FAKE judge for harness tests: case-insensitive substring match, no LLM call."""

    kind = "fake"

    async def judge_text(
        self, question: Question, answer: SystemAnswer, *, tags: dict[str, Any]
    ) -> TextVerdict:
        correct = bool(answer.answer.strip()) and _fold(question.expected.answer) in _fold(
            answer.answer
        )
        return TextVerdict(reason="fake judge: substring match", correct=correct)

    async def judge_key_points(
        self, question: Question, answer: SystemAnswer, *, tags: dict[str, Any]
    ) -> KeyPointVerdict:
        text = _fold(answer.answer)
        covered = [_fold(kp) in text for kp in question.key_points]
        return KeyPointVerdict(reason="fake judge: substring match", covered=covered)
