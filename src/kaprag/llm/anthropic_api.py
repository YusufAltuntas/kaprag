"""`anthropic_api` backend: the Messages API with an API key (for external demos)."""

from __future__ import annotations

import os
from typing import Any

import anthropic

from kaprag.llm.types import BackendError, BackendResponse, TransientBackendError, Usage


class AnthropicAPIBackend:
    name = "anthropic_api"

    def __init__(self, *, client: Any | None = None, max_tokens: int = 16000) -> None:
        self._client = client
        self._max_tokens = max_tokens

    def _get_client(self) -> Any:
        if self._client is None:
            key = os.environ.get("ANTHROPIC_API_KEY")
            if not key:
                raise BackendError("ANTHROPIC_API_KEY is not set (required for anthropic_api)")
            # Retries are handled by LLMClient so they are counted and traced.
            self._client = anthropic.AsyncAnthropic(api_key=key, max_retries=0)
        return self._client

    async def generate(self, system: str, user: str, model: str) -> BackendResponse:
        client = self._get_client()
        try:
            response = await client.messages.create(
                model=model,
                max_tokens=self._max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
        except (
            anthropic.RateLimitError,  # 429
            anthropic.InternalServerError,  # >=500, incl. OverloadedError (529)
            anthropic.APIConnectionError,  # network errors and timeouts
        ) as exc:
            raise TransientBackendError(f"Anthropic API transient error: {exc}") from exc
        except anthropic.APIStatusError as exc:
            raise BackendError(f"Anthropic API error: {exc}") from exc

        if response.stop_reason == "refusal":
            raise BackendError("Anthropic API refused the request (stop_reason=refusal)")
        text = "".join(block.text for block in response.content if block.type == "text")
        return BackendResponse(text=text, usage=Usage.from_mapping(response.usage))
