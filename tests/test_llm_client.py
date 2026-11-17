"""LLMClient behaviour (cache, JSON validation, retry, tracing) with a fake backend."""

import asyncio
import json

import pytest
from pydantic import BaseModel

from kaprag.llm import LLMCallError, LLMClient, TransientBackendError, Usage
from kaprag.llm.types import BackendError, BackendResponse

MODEL = "claude-haiku-5-5"


class Answer(BaseModel):
    answer: str
    confidence: float


class FakeBackend:
    """Replays scripted outcomes (text or exception) and records every request."""

    def __init__(self, outcomes, name="fake"):
        self.name = name
        self.outcomes = list(outcomes)
        self.requests = []

    async def generate(self, system, user, model):
        self.requests.append((system, user, model))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return BackendResponse(text=outcome, usage=Usage(input_tokens=10, output_tokens=5))


@pytest.fixture
def make_client(tmp_dir):
    sleeps = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    def make(backend, **kwargs):
        client = LLMClient(
            backend,
            cache_dir=tmp_dir / "cache",
            trace_path=tmp_dir / "traces.jsonl",
            sleep=fake_sleep,
            **kwargs,
        )
        client.sleeps = sleeps
        return client

    return make


def read_traces(client):
    lines = client.trace.path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def run(coro):
    return asyncio.run(coro)


# --- cache -----------------------------------------------------------------


def test_cache_hit_returns_cached_and_skips_backend(make_client):
    backend = FakeBackend(["hello"])
    client = make_client(backend)

    first = run(client.complete("sys", "user", MODEL))
    second = run(client.complete("sys", "user", MODEL))

    assert (first.text, first.cached, first.attempts) == ("hello", False, 1)
    assert (second.text, second.cached, second.attempts) == ("hello", True, 0)
    assert second.usage == first.usage
    assert len(backend.requests) == 1


@pytest.mark.parametrize(
    "change",
    ["backend", "model", "system", "user", "schema"],
)
def test_cache_key_changes_with_each_input(make_client, tmp_dir, change):
    base = {"system": "sys", "user": "user", "model": MODEL, "schema": None}
    valid_json = '{"answer": "a", "confidence": 1}'
    run(make_client(FakeBackend([valid_json])).complete(**base))

    varied = dict(base)
    backend_name = "fake"
    if change == "backend":
        backend_name = "other"
    elif change == "model":
        varied["model"] = "claude-sonnet-5-5"
    elif change == "schema":
        varied["schema"] = Answer
    else:
        varied[change] = base[change] + "!"
    backend = FakeBackend([valid_json], name=backend_name)
    result = run(make_client(backend).complete(**varied))

    assert result.cached is False
    assert len(backend.requests) == 1


def test_failed_call_is_not_cached(make_client):
    backend = FakeBackend([BackendError("boom"), "ok"])
    client = make_client(backend)

    with pytest.raises(LLMCallError):
        run(client.complete("sys", "user", MODEL))
    result = run(client.complete("sys", "user", MODEL))

    assert (result.text, result.cached) == ("ok", False)


def test_stale_cache_entry_that_fails_validation_is_a_miss_and_is_replaced(make_client, tmp_dir):
    old, new = '{"answer": "old", "confidence": 1}', '{"answer": "new", "confidence": 0.5}'
    backend = FakeBackend([old, new])
    client = make_client(backend)
    run(client.complete("sys", "user", MODEL, schema=Answer))
    [entry] = (tmp_dir / "cache").rglob("*.json")
    stale = json.loads(entry.read_text(encoding="utf-8"))
    stale["text"] = '{"answer": "old"}'  # e.g. written before the schema gained a field
    entry.write_text(json.dumps(stale), encoding="utf-8")

    result = run(client.complete("sys", "user", MODEL, schema=Answer))

    assert (result.parsed, result.cached, result.attempts) == (
        Answer(answer="new", confidence=0.5),
        False,
        1,
    )
    assert len(backend.requests) == 2
    assert json.loads(entry.read_text(encoding="utf-8"))["text"] == new
    again = run(client.complete("sys", "user", MODEL, schema=Answer))
    assert (again.parsed.answer, again.cached) == ("new", True)
    assert len(backend.requests) == 2


def test_cache_can_be_disabled(tmp_dir):
    backend = FakeBackend(["a", "b"])
    client = LLMClient(backend, cache_dir=None, trace_path=tmp_dir / "t.jsonl")

    assert run(client.complete("s", "u", MODEL)).text == "a"
    assert run(client.complete("s", "u", MODEL)).text == "b"


# --- JSON validation ---------------------------------------------------------


def test_valid_json_is_parsed_and_schema_is_in_prompt(make_client):
    backend = FakeBackend(['```json\n{"answer": "Ankara", "confidence": 0.9}\n```'])
    result = run(make_client(backend).complete("sys", "user", MODEL, schema=Answer))

    assert result.parsed == Answer(answer="Ankara", confidence=0.9)
    sent_user = backend.requests[0][1]
    assert sent_user.startswith("user")
    assert '"confidence"' in sent_user  # JSON Schema appended to the prompt


def test_invalid_then_valid_json_retries_once(make_client):
    backend = FakeBackend(["not json", '{"answer": "x", "confidence": 0.5}'])
    result = run(make_client(backend).complete("sys", "user", MODEL, schema=Answer))

    assert result.parsed == Answer(answer="x", confidence=0.5)
    assert result.attempts == 2
    assert result.usage == Usage(input_tokens=20, output_tokens=10)  # both replies counted
    assert "Validation error" in backend.requests[1][1]


def test_invalid_json_twice_raises_and_is_traced(make_client):
    backend = FakeBackend(["nope", '{"answer": "x"}'])
    client = make_client(backend)

    with pytest.raises(LLMCallError, match="invalid JSON"):
        run(client.complete("sys", "user", MODEL, schema=Answer, tags={"qid": "q1"}))

    [record] = read_traces(client)
    assert record["error"].startswith("invalid JSON")
    assert record["attempts"] == 2
    assert record["tags"] == {"qid": "q1"}
    assert len(backend.requests) == 2


def test_no_schema_means_parsed_is_none(make_client):
    result = run(make_client(FakeBackend(["plain"])).complete("s", "u", MODEL))
    assert result.parsed is None


# --- retry / backoff ----------------------------------------------------------


def test_transient_errors_back_off_exponentially_then_succeed(make_client):
    backend = FakeBackend([TransientBackendError("429"), TransientBackendError("429"), "ok"])
    client = make_client(backend, backoff_base_s=2.0)

    result = run(client.complete("s", "u", MODEL))

    assert (result.text, result.attempts) == ("ok", 3)
    assert len(client.sleeps) == 2
    assert 2.0 <= client.sleeps[0] < 2.3
    assert 4.0 <= client.sleeps[1] < 4.5


def test_three_transient_failures_raise_call_error_and_trace(make_client):
    backend = FakeBackend([TransientBackendError("429")] * 3)
    client = make_client(backend)

    with pytest.raises(LLMCallError) as excinfo:
        run(client.complete("s", "u", MODEL))

    assert excinfo.value.attempts == 3
    [record] = read_traces(client)
    assert record["attempts"] == 3
    assert "429" in record["error"]
    assert record["cached"] is False


def test_non_transient_error_is_not_retried(make_client):
    backend = FakeBackend([BackendError("bad request"), "never"])
    client = make_client(backend)

    with pytest.raises(LLMCallError):
        run(client.complete("s", "u", MODEL))
    assert len(backend.requests) == 1
    assert client.sleeps == []


def test_unknown_model_is_rejected_before_any_request(make_client):
    backend = FakeBackend(["x"])
    with pytest.raises(ValueError, match="no price"):
        run(make_client(backend).complete("s", "u", "claude-unknown"))
    assert backend.requests == []


# --- concurrency ----------------------------------------------------------------


def test_concurrency_is_limited_by_semaphore(make_client):
    active = 0
    peak = 0

    class SlowBackend:
        name = "slow"

        async def generate(self, system, user, model):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.01)
            active -= 1
            return BackendResponse(text=user, usage=Usage())

    client = make_client(SlowBackend(), max_concurrency=2)

    async def many():
        return await asyncio.gather(*(client.complete("s", f"u{i}", MODEL) for i in range(6)))

    results = run(many())
    assert [r.text for r in results] == [f"u{i}" for i in range(6)]
    assert peak == 2


# --- tracing ----------------------------------------------------------------------


def test_trace_records_success_and_cache_hit(make_client):
    client = make_client(FakeBackend(["hi"]))
    run(client.complete("s", "u", MODEL, tags={"qid": "q7", "step": "answer"}))
    run(client.complete("s", "u", MODEL))

    first, second = read_traces(client)
    assert first["backend"] == "fake"
    assert first["model"] == MODEL
    assert len(first["prompt_hash"]) == 64
    assert (first["input_tokens"], first["output_tokens"]) == (10, 5)
    assert first["cached"] is False
    assert first["attempts"] == 1
    assert first["error"] is None
    assert first["cost_usd"] == pytest.approx((10 * 0.10 + 5 * 0.50) / 1e6)
    assert first["tags"] == {"qid": "q7", "step": "answer"}
    assert first["latency_ms"] >= 0
    assert "ts" in first

    assert second["cached"] is True
    assert second["prompt_hash"] == first["prompt_hash"]
    assert second["source_latency_ms"] is not None
    assert second["tags"] == {}


def test_trace_lines_stay_intact_under_concurrency(make_client):
    n = 40
    client = make_client(FakeBackend([f"r{i}" for i in range(n)]), max_concurrency=8)

    async def many():
        await asyncio.gather(*(client.complete("s", f"u{i}", MODEL) for i in range(n)))

    run(many())
    assert len(read_traces(client)) == n


def test_complete_sync_runs_repeatedly(make_client):
    client = make_client(FakeBackend(["a", "b"]))
    assert client.complete_sync("s", "u1", MODEL).text == "a"
    assert client.complete_sync("s", "u2", MODEL).text == "b"
