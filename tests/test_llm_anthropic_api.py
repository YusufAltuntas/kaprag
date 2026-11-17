"""anthropic_api backend against a stubbed SDK client (no key, no network)."""

import asyncio
from types import SimpleNamespace

import anthropic
import httpx2
import pytest

from kaprag.llm.anthropic_api import AnthropicAPIBackend
from kaprag.llm.types import BackendError, TransientBackendError, Usage


class StubMessages:
    def __init__(self, outcome):
        self.outcome = outcome
        self.kwargs = None

    async def create(self, **kwargs):
        self.kwargs = kwargs
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def stub_client(outcome):
    return SimpleNamespace(messages=StubMessages(outcome))


def status_error(cls, status):
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx2.Response(status, request=request)
    return cls("error", response=response, body=None)


def generate(backend):
    return asyncio.run(backend.generate("sys", "user", "claude-haiku-5-5"))


def test_request_shape_and_response_mapping():
    message = SimpleNamespace(
        stop_reason="end_turn",
        content=[
            SimpleNamespace(type="thinking", thinking=""),
            SimpleNamespace(type="text", text="Ankara"),
        ],
        usage=SimpleNamespace(
            input_tokens=12,
            output_tokens=3,
            cache_read_input_tokens=None,
            cache_creation_input_tokens=None,
        ),
    )
    client = stub_client(message)
    response = generate(AnthropicAPIBackend(client=client, max_tokens=500))

    assert client.messages.kwargs == {
        "model": "claude-haiku-5-5",
        "max_tokens": 500,
        "system": "sys",
        "messages": [{"role": "user", "content": "user"}],
    }
    assert response.text == "Ankara"
    assert response.usage == Usage(input_tokens=12, output_tokens=3)


@pytest.mark.parametrize(
    "error",
    [
        status_error(anthropic.RateLimitError, 429),
        status_error(anthropic.InternalServerError, 500),
        status_error(anthropic.OverloadedError, 529),
    ],
)
def test_rate_limit_and_server_errors_are_transient(error):
    with pytest.raises(TransientBackendError):
        generate(AnthropicAPIBackend(client=stub_client(error)))


def test_bad_request_is_not_transient():
    error = status_error(anthropic.BadRequestError, 400)
    with pytest.raises(BackendError) as excinfo:
        generate(AnthropicAPIBackend(client=stub_client(error)))
    assert not isinstance(excinfo.value, TransientBackendError)


def test_missing_key_is_a_backend_error(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(BackendError, match="ANTHROPIC_API_KEY"):
        generate(AnthropicAPIBackend())
