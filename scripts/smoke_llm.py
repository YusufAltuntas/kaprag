"""Smoke test: one claude-haiku-5-5 call through the claude_code backend (subscription).

Run: uv run python scripts/smoke_llm.py
A second run is served from the disk cache (cached=True) and spends no quota.
"""

import sys

from kaprag.llm import create_client

SYSTEM = "You are a terse assistant. Answer in one short sentence."
USER = "Türkiye'nin başkenti neresidir?"


def main() -> None:
    # Piped stdout on Windows defaults to the locale code page; the answer is Turkish.
    sys.stdout.reconfigure(encoding="utf-8")
    client = create_client("claude_code")
    result = client.complete_sync(SYSTEM, USER, "claude-haiku-5-5", tags={"script": "smoke_llm"})
    print(f"text: {result.text}")
    print(f"usage: {result.usage}")
    print(f"latency_ms={result.latency_ms:.0f} cached={result.cached} attempts={result.attempts}")
    print(f"trace: {client.trace.path}")


if __name__ == "__main__":
    main()
