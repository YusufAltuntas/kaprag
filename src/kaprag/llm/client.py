"""`LLMClient`: the only entry point for LLM calls in kaprag.

Adds, on top of a backend: a disk cache, a concurrency semaphore, retry with
exponential backoff on transient errors, JSON-schema validation with one retry,
and a JSONL trace record for every call (cache hits and failures included).
"""

from __future__ import annotations

import asyncio
import os
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ValidationError

from kaprag.llm import pricing
from kaprag.llm.cache import DiskCache, cache_key
from kaprag.llm.structured import parse_output, retry_instruction, schema_instruction
from kaprag.llm.trace import TraceWriter
from kaprag.llm.types import (
    BackendError,
    BackendResponse,
    LLMCallError,
    LLMResult,
    TransientBackendError,
    Usage,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CACHE_DIR = REPO_ROOT / ".cache" / "llm"
DEFAULT_TRACE_PATH = REPO_ROOT / ".cache" / "llm-traces.jsonl"

MAX_ATTEMPTS = 3  # per backend request, on transient errors
MAX_JSON_RETRIES = 1


class Backend(Protocol):
    name: str

    async def generate(self, system: str, user: str, model: str) -> BackendResponse: ...


Sleep = Callable[[float], Awaitable[None]]


class LLMClient:
    """Thin client over one backend. Create it with `create_client()` in app code."""

    def __init__(
        self,
        backend: Backend,
        *,
        max_concurrency: int = 4,
        cache_dir: Path | None = DEFAULT_CACHE_DIR,
        trace_path: Path = DEFAULT_TRACE_PATH,
        backoff_base_s: float = 2.0,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        """`cache_dir=None` disables the cache; `sleep` is injectable for tests."""
        self.backend = backend
        self.max_concurrency = max_concurrency
        self.cache = DiskCache(cache_dir) if cache_dir is not None else None
        self.trace = TraceWriter(trace_path)
        self._backoff_base_s = backoff_base_s
        self._sleep = sleep
        self._semaphore: asyncio.Semaphore | None = None
        self._semaphore_loop: asyncio.AbstractEventLoop | None = None

    def _get_semaphore(self) -> asyncio.Semaphore:
        # A semaphore is bound to one event loop; complete_sync runs a new loop per call.
        loop = asyncio.get_running_loop()
        if self._semaphore is None or self._semaphore_loop is not loop:
            self._semaphore = asyncio.Semaphore(self.max_concurrency)
            self._semaphore_loop = loop
        return self._semaphore

    async def complete(
        self,
        system: str,
        user: str,
        model: str,
        schema: type[BaseModel] | None = None,
        *,
        tags: dict[str, Any] | None = None,
    ) -> LLMResult:
        """Run one completion.

        With `schema`, the JSON Schema is appended to the user prompt and the reply
        is validated into `LLMResult.parsed`; an invalid reply is retried once.
        `tags` are copied verbatim into the trace record (e.g. question id, step).

        Raises `LLMCallError` when the call fails for good (transient errors after
        3 attempts, a non-retryable backend error, or invalid JSON twice). The
        failure is traced first, so a batch can catch it, mark the item as an
        error and move on. Failed calls are never cached. A cached reply that no longer
        validates against `schema` is treated as a miss and overwritten by the new reply.
        """
        if model not in pricing.PRICES:
            raise ValueError(f"no price for model {model!r}; add it to kaprag.llm.pricing")
        start = time.perf_counter()
        schema_json = schema.model_json_schema() if schema is not None else None
        key = cache_key(
            backend=self.backend.name, model=model, system=system, user=user, schema=schema_json
        )

        hit = self.cache.get(key) if self.cache is not None else None
        parsed: BaseModel | None = None
        if hit is not None and schema is not None:
            try:
                parsed = parse_output(hit["text"], schema)
            except ValidationError:
                hit = None  # stale entry: call the backend again and overwrite it
        if hit is not None:
            result = LLMResult(
                text=hit["text"],
                parsed=parsed,
                usage=hit["usage"],
                latency_ms=_elapsed_ms(start),
                cached=True,
                attempts=0,
            )
            self._write_trace(key, model, result, tags=tags, source_latency_ms=hit["latency_ms"])
            return result

        usage = Usage()
        attempts = 0
        prompt = user + schema_instruction(schema) if schema is not None else user
        try:
            for json_try in range(MAX_JSON_RETRIES + 1):
                response, n = await self._generate_with_retry(system, prompt, model)
                attempts += n
                usage = usage + response.usage
                if schema is None:
                    parsed = None
                    break
                try:
                    parsed = parse_output(response.text, schema)
                    break
                except ValidationError as exc:
                    if json_try == MAX_JSON_RETRIES:
                        raise LLMCallError(
                            f"invalid JSON for {schema.__name__} after "
                            f"{MAX_JSON_RETRIES + 1} replies: {exc}",
                            attempts=attempts,
                            usage=usage,
                        ) from exc
                    prompt = prompt + retry_instruction(str(exc))
        except _AttemptsError as exc:
            attempts += exc.attempts
            error = LLMCallError(str(exc.cause), attempts=attempts, usage=usage)
            self._write_failure(key, model, start, error, tags)
            raise error from exc.cause
        except LLMCallError as exc:
            self._write_failure(key, model, start, exc, tags)
            raise

        result = LLMResult(
            text=response.text,
            parsed=parsed,
            usage=usage,
            latency_ms=_elapsed_ms(start),
            cached=False,
            attempts=attempts,
        )
        if self.cache is not None:
            self.cache.put(key, text=response.text, usage=usage, latency_ms=result.latency_ms)
        self._write_trace(key, model, result, tags=tags)
        return result

    def complete_sync(
        self,
        system: str,
        user: str,
        model: str,
        schema: type[BaseModel] | None = None,
        *,
        tags: dict[str, Any] | None = None,
    ) -> LLMResult:
        """Blocking wrapper around `complete` for scripts (not for use inside a loop)."""
        return asyncio.run(self.complete(system, user, model, schema, tags=tags))

    async def _generate_with_retry(
        self, system: str, user: str, model: str
    ) -> tuple[BackendResponse, int]:
        """Return (response, attempts); raise _AttemptsError when giving up."""
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                async with self._get_semaphore():
                    return await self.backend.generate(system, user, model), attempt
            except TransientBackendError as exc:
                if attempt == MAX_ATTEMPTS:
                    raise _AttemptsError(exc, attempt) from exc
                delay = self._backoff_base_s * 2 ** (attempt - 1)
                await self._sleep(delay * (1 + random.random() * 0.1))
            except BackendError as exc:
                raise _AttemptsError(exc, attempt) from exc
        raise AssertionError("unreachable")

    def _write_failure(
        self, key: str, model: str, start: float, error: LLMCallError, tags: dict[str, Any] | None
    ) -> None:
        failed = LLMResult(
            text="",
            parsed=None,
            usage=error.usage,
            latency_ms=_elapsed_ms(start),
            cached=False,
            attempts=error.attempts,
        )
        self._write_trace(key, model, failed, tags=tags, error=str(error))

    def _write_trace(
        self,
        key: str,
        model: str,
        result: LLMResult,
        *,
        tags: dict[str, Any] | None,
        error: str | None = None,
        source_latency_ms: float | None = None,
    ) -> None:
        self.trace.write(
            {
                "ts": datetime.now(UTC).isoformat(),
                "backend": self.backend.name,
                "model": model,
                "prompt_hash": key,
                **asdict(result.usage),
                "latency_ms": round(result.latency_ms, 1),
                "source_latency_ms": (
                    round(source_latency_ms, 1) if source_latency_ms is not None else None
                ),
                "cached": result.cached,
                "attempts": result.attempts,
                "error": error,
                "cost_usd": pricing.cost_usd(model, result.usage),
                "tags": tags or {},
            }
        )


class _AttemptsError(Exception):
    """Internal: a backend request gave up after `attempts` tries."""

    def __init__(self, cause: BackendError, attempts: int) -> None:
        super().__init__(str(cause))
        self.cause = cause
        self.attempts = attempts


def _elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000


def create_client(backend: str | None = None, **kwargs: Any) -> LLMClient:
    """Build a client from the environment.

    `backend` defaults to `LLM_BACKEND` (else `claude_code`); concurrency defaults to
    `LLM_MAX_CONCURRENCY` (else 4). Extra kwargs go to `LLMClient`.
    """
    from kaprag.llm.anthropic_api import AnthropicAPIBackend
    from kaprag.llm.claude_code import ClaudeCodeBackend

    name = backend or os.environ.get("LLM_BACKEND", "claude_code")
    backends: dict[str, Callable[[], Backend]] = {
        "claude_code": ClaudeCodeBackend,
        "anthropic_api": AnthropicAPIBackend,
    }
    if name not in backends:
        raise ValueError(f"unknown LLM_BACKEND {name!r}; expected one of {sorted(backends)}")
    kwargs.setdefault("max_concurrency", int(os.environ.get("LLM_MAX_CONCURRENCY", "4")))
    return LLMClient(backends[name](), **kwargs)
