"""`claude_code` backend: the Claude subscription via the Claude Agent SDK.

Every request runs as an isolated single-turn completion: no tools, a custom
system prompt (not the Claude Code preset), no filesystem settings or CLAUDE.md,
no MCP servers, and a working directory in the system temp folder (outside the
repo). `ANTHROPIC_API_KEY` is removed so usage is billed to the subscription.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import claude_agent_sdk
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKError,
    ResultMessage,
    TextBlock,
)

from kaprag.llm.types import BackendError, BackendResponse, TransientBackendError, Usage

QueryFn = Callable[..., AsyncIterator[Any]]

# HTTP statuses / assistant error kinds that mean "back off and try again".
_TRANSIENT_STATUSES = {429, 500, 502, 503, 504, 529}
_TRANSIENT_KINDS = {"rate_limit", "server_error"}

SCRUBBED_ENV_VARS = ("ANTHROPIC_API_KEY",)


def default_cwd() -> Path:
    path = Path(tempfile.gettempdir()) / "kaprag-claude-code"
    path.mkdir(parents=True, exist_ok=True)
    return path


class ClaudeCodeBackend:
    name = "claude_code"

    def __init__(
        self,
        *,
        query_fn: QueryFn | None = None,
        cwd: Path | None = None,
        cli_path: str | Path | None = None,
    ) -> None:
        self._query_fn = query_fn or claude_agent_sdk.query
        self._cwd = cwd
        self._cli_path = cli_path

    def build_options(self, system: str, model: str) -> ClaudeAgentOptions:
        return ClaudeAgentOptions(
            system_prompt=system,
            model=model,
            tools=[],
            max_turns=1,
            setting_sources=[],
            strict_mcp_config=True,
            verbatim_prompts=True,
            cwd=self._cwd or default_cwd(),
            cli_path=self._cli_path,
        )

    async def generate(self, system: str, user: str, model: str) -> BackendResponse:
        # The SDK merges the parent environment into the CLI subprocess and offers
        # no way to drop a variable, so remove it from this process instead. In
        # claude_code mode the key must not be set at all (see CLAUDE.md).
        for var in SCRUBBED_ENV_VARS:
            os.environ.pop(var, None)

        texts: list[str] = []
        error_kind: str | None = None
        result: ResultMessage | None = None
        try:
            async for message in self._query_fn(
                prompt=user, options=self.build_options(system, model)
            ):
                if isinstance(message, AssistantMessage):
                    if message.error:
                        error_kind = message.error
                    if message.parent_tool_use_id is None:
                        texts.extend(b.text for b in message.content if isinstance(b, TextBlock))
                elif isinstance(message, ResultMessage):
                    result = message
        except ClaudeSDKError as exc:
            raise BackendError(f"Claude Agent SDK error: {exc}") from exc

        if result is None:
            raise BackendError("Claude Code returned no result message")
        if result.is_error or error_kind:
            detail = (
                f"subtype={result.subtype} status={result.api_error_status} "
                f"error={error_kind} errors={result.errors} result={result.result!r}"
            )
            if result.api_error_status in _TRANSIENT_STATUSES or error_kind in _TRANSIENT_KINDS:
                raise TransientBackendError(f"Claude Code transient error: {detail}")
            raise BackendError(f"Claude Code error: {detail}")

        text = result.result if result.result is not None else "".join(texts)
        return BackendResponse(text=text, usage=Usage.from_mapping(result.usage))
