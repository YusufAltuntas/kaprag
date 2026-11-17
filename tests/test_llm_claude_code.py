"""claude_code backend: options, env scrubbing, result/error mapping (no real CLI call)."""

import asyncio
import dataclasses
import sys
import tempfile
from pathlib import Path

import anyio
import pytest
from claude_agent_sdk import (
    AssistantMessage,
    ProcessError,
    RateLimitEvent,
    RateLimitInfo,
    ResultError,
    ResultMessage,
    TextBlock,
)

from kaprag.llm import LLMCallError, LLMClient, TransientBackendError
from kaprag.llm.claude_code import ClaudeCodeBackend
from kaprag.llm.types import BackendError, Usage

REPO_ROOT = Path(__file__).resolve().parents[1]


def result_message(**overrides):
    fields = {
        "subtype": "success",
        "duration_ms": 10,
        "duration_api_ms": 8,
        "is_error": False,
        "num_turns": 1,
        "session_id": "s",
        "result": "Ankara",
        "usage": {
            "input_tokens": 3,
            "output_tokens": 7,
            "cache_read_input_tokens": 100,
            "cache_creation_input_tokens": 50,
        },
    }
    fields.update(overrides)
    return ResultMessage(**fields)


def scripted_query(*runs, raise_on_error_result=True):
    """A stand-in for claude_agent_sdk.query that replays one message list per call.

    Like the real SDK (claude_agent_sdk/_internal/query.py), a run whose last
    result has is_error=True yields that ResultMessage and then raises
    ResultError carrying the result payload, because the CLI exits non-zero.
    A run may also end with an exception instance, which is raised as-is.
    """
    calls = []
    runs = list(runs)

    async def query(*, prompt, options):
        calls.append((prompt, options))
        last_result = None
        for item in runs.pop(0):
            if isinstance(item, Exception):
                raise item
            if isinstance(item, ResultMessage):
                last_result = item
            yield item
        if raise_on_error_result and last_result is not None and last_result.is_error:
            data = {"type": "result", **dataclasses.asdict(last_result)}
            raise ResultError(
                f"Claude Code returned an error result: {last_result.result}",
                data=data,
                exit_code=1,
            )

    query.calls = calls
    return query


def error_result(**overrides):
    fields = {"is_error": True, "result": "API Error", "usage": None}
    fields.update(overrides)
    return result_message(**fields)


def rejected_limit_event():
    info = RateLimitInfo(status="rejected", resets_at=1_900_000_000, rate_limit_type="five_hour")
    return RateLimitEvent(rate_limit_info=info, uuid="u", session_id="s")


def generate(backend):
    return asyncio.run(backend.generate("system text", "user text", "claude-haiku-5-5"))


def test_options_isolate_the_call():
    query = scripted_query([result_message()])
    generate(ClaudeCodeBackend(query_fn=query))

    prompt, options = query.calls[0]
    assert prompt == "user text"
    assert options.system_prompt == "system text"  # custom string, not the preset
    assert options.model == "claude-haiku-5-5"
    assert options.tools == []
    assert options.max_turns == 1
    assert options.setting_sources == []  # no user/project settings, no CLAUDE.md
    assert options.strict_mcp_config is True
    cwd = Path(options.cwd).resolve()
    assert cwd.is_relative_to(Path(tempfile.gettempdir()).resolve())
    assert not cwd.is_relative_to(REPO_ROOT)


def test_success_maps_text_and_usage():
    query = scripted_query([result_message()])
    response = generate(ClaudeCodeBackend(query_fn=query))

    assert response.text == "Ankara"
    assert response.usage == Usage(
        input_tokens=3, output_tokens=7, cache_read_input_tokens=100, cache_creation_input_tokens=50
    )


def test_text_falls_back_to_assistant_blocks():
    assistant = AssistantMessage(content=[TextBlock(text="Ank"), TextBlock(text="ara")], model="m")
    query = scripted_query([assistant, result_message(result=None)])
    assert generate(ClaudeCodeBackend(query_fn=query)).text == "Ankara"


@pytest.mark.parametrize(
    "run",
    [
        pytest.param([error_result(api_error_status=429)], id="http-429"),
        pytest.param([error_result(api_error_status=529)], id="http-529-overloaded"),
        pytest.param([error_result(api_error_status=503)], id="http-503"),
        pytest.param(
            [AssistantMessage(content=[], model="m", error="rate_limit"), error_result()],
            id="assistant-rate-limit",
        ),
        pytest.param(
            [AssistantMessage(content=[], model="m", error="server_error"), error_result()],
            id="assistant-server-error",
        ),
        pytest.param(
            [rejected_limit_event(), error_result(result="usage limit reached")],
            id="subscription-limit-rejected",
        ),
        pytest.param([error_result(terminal_reason="api_error")], id="api-error-no-status"),
    ],
)
def test_transient_failures_raise_transient_error(run):
    with pytest.raises(TransientBackendError) as excinfo:
        generate(ClaudeCodeBackend(query_fn=scripted_query(run)))
    assert isinstance(excinfo.value.__cause__, ResultError)


@pytest.mark.parametrize(
    "run",
    [
        pytest.param([error_result(api_error_status=400)], id="http-400"),
        pytest.param(
            [error_result(api_error_status=400, terminal_reason="api_error")],
            id="api-error-400",
        ),
        pytest.param(
            [AssistantMessage(content=[], model="m", error="billing_error"), error_result()],
            id="billing-error",
        ),
        pytest.param(
            [
                AssistantMessage(content=[], model="m", error="authentication_failed"),
                error_result(terminal_reason="api_error"),
            ],
            id="auth-failed",
        ),
        pytest.param(
            [error_result(subtype="error_max_turns", terminal_reason="max_turns")],
            id="max-turns",
        ),
        pytest.param([ProcessError("crashed", exit_code=1)], id="process-crash"),
    ],
)
def test_other_failures_are_not_transient(run):
    with pytest.raises(BackendError) as excinfo:
        generate(ClaudeCodeBackend(query_fn=scripted_query(run)))
    assert not isinstance(excinfo.value, TransientBackendError)


def test_error_result_without_exception_is_still_classified():
    query = scripted_query([error_result(api_error_status=429)], raise_on_error_result=False)
    with pytest.raises(TransientBackendError):
        generate(ClaudeCodeBackend(query_fn=query))


def test_client_retries_rate_limit_through_backend(tmp_dir):
    limited = [error_result(api_error_status=429)]
    query = scripted_query(
        limited, [rejected_limit_event(), error_result()], [result_message(result="ok")]
    )
    sleeps = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    client = LLMClient(
        ClaudeCodeBackend(query_fn=query),
        cache_dir=tmp_dir / "cache",
        trace_path=tmp_dir / "t.jsonl",
        sleep=fake_sleep,
    )
    result = asyncio.run(client.complete("s", "u", "claude-haiku-5-5"))

    assert (result.text, result.attempts, len(sleeps)) == ("ok", 3, 2)


def test_client_gives_up_after_three_rate_limits(tmp_dir):
    limited = [error_result(api_error_status=429)]

    async def no_sleep(seconds):
        pass

    client = LLMClient(
        ClaudeCodeBackend(query_fn=scripted_query(limited, limited, limited)),
        cache_dir=tmp_dir / "cache",
        trace_path=tmp_dir / "t.jsonl",
        sleep=no_sleep,
    )
    with pytest.raises(LLMCallError):
        asyncio.run(client.complete("s", "u", "claude-haiku-5-5"))


def test_api_key_is_removed_from_subprocess_env(monkeypatch):
    """Drive the real SDK transport up to process spawn and inspect its env."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-not-a-real-key")
    captured = {}

    class SpawnBlocked(Exception):
        pass

    async def fake_open_process(cmd, **kwargs):
        captured.update(kwargs)
        raise SpawnBlocked

    monkeypatch.setattr(anyio, "open_process", fake_open_process)
    # Any existing non-batch file works as cli_path: the spawn is intercepted.
    backend = ClaudeCodeBackend(cli_path=sys.executable)

    with pytest.raises(Exception):  # noqa: B017 - the SDK wraps the spawn error
        generate(backend)

    assert "env" in captured, "transport never reached process spawn"
    assert "ANTHROPIC_API_KEY" not in captured["env"]
    assert captured["env"]["PATH"]  # the rest of the environment is still passed
