"""Shared value types and errors for the LLM client."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Usage:
    """Token counts reported by a backend for one logical call."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_input_tokens=self.cache_read_input_tokens + other.cache_read_input_tokens,
            cache_creation_input_tokens=(
                self.cache_creation_input_tokens + other.cache_creation_input_tokens
            ),
        )

    @classmethod
    def from_mapping(cls, data: Any) -> Usage:
        """Build from a usage dict or an SDK usage object; missing fields count as 0."""

        def get(name: str) -> int:
            value = data.get(name) if isinstance(data, dict) else getattr(data, name, None)
            return int(value or 0)

        if data is None:
            return cls()
        return cls(
            input_tokens=get("input_tokens"),
            output_tokens=get("output_tokens"),
            cache_read_input_tokens=get("cache_read_input_tokens"),
            cache_creation_input_tokens=get("cache_creation_input_tokens"),
        )


@dataclass(frozen=True)
class BackendResponse:
    """Raw text and usage returned by a single backend request."""

    text: str
    usage: Usage


@dataclass(frozen=True)
class LLMResult:
    """Result of `LLMClient.complete`.

    `parsed` is the validated pydantic object when a schema was given, else None.
    `latency_ms` is the wall time of this call (a cache lookup on a hit).
    `attempts` counts backend requests made (0 on a cache hit).
    """

    text: str
    parsed: Any
    usage: Usage
    latency_ms: float
    cached: bool
    attempts: int = 0


class LLMError(Exception):
    """Base class for all errors raised by `kaprag.llm`."""


class BackendError(LLMError):
    """A backend request failed and retrying it will not help."""


class TransientBackendError(BackendError):
    """A backend request failed due to rate limits, quota or overload; safe to retry."""


class LLMCallError(LLMError):
    """A `complete` call failed for good (retries exhausted or invalid JSON twice).

    The failure has already been written to the trace; batch callers should catch
    this, record the item as an error and continue.
    """

    def __init__(self, message: str, *, attempts: int, usage: Usage) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.usage = usage
