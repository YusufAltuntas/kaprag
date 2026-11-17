"""API list prices: the single source of truth for API-equivalent cost.

The Claude subscription shows no dollars, so cost is always computed as
tokens x API list price, whichever backend served the call.
"""

from __future__ import annotations

from dataclasses import dataclass

from kaprag.llm.types import Usage


@dataclass(frozen=True)
class ModelPrice:
    """USD per one million tokens."""

    input_per_mtok: float
    output_per_mtok: float


# API list prices as of 2026-11.
PRICES: dict[str, ModelPrice] = {
    "claude-haiku-5-5": ModelPrice(input_per_mtok=0.10, output_per_mtok=0.50),
    "claude-sonnet-5-5": ModelPrice(input_per_mtok=2.00, output_per_mtok=10.00),
    "claude-opus-5-5": ModelPrice(input_per_mtok=4.00, output_per_mtok=20.00),
}


def cost_usd(model: str, usage: Usage) -> float:
    """API-equivalent cost of `usage` on `model`.

    Prompt-cache read/write tokens are priced as plain input tokens: Claude Code
    caches its prompt on its own (a short call reports ~2 input tokens and
    hundreds of cache-write tokens), while the anthropic_api backend does not use
    prompt caching. Pricing every prompt token at the input rate keeps the cost of
    the same prompt comparable across backends and runs.
    Raises KeyError for a model without a price, so a missing price is never
    silently reported as $0.
    """
    price = PRICES[model]
    prompt_tokens = (
        usage.input_tokens + usage.cache_read_input_tokens + usage.cache_creation_input_tokens
    )
    return (
        prompt_tokens * price.input_per_mtok + usage.output_tokens * price.output_per_mtok
    ) / 1_000_000
