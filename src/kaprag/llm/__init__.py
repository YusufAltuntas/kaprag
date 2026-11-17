"""LLM access for kaprag. All LLM SDK calls in the codebase go through this package.

Usage:
    from kaprag.llm import create_client
    client = create_client()  # backend from LLM_BACKEND
    result = await client.complete(system, user, "claude-haiku-5-5", schema=MyModel,
                                   tags={"qid": "q001", "step": "judge"})
"""

from kaprag.llm.client import LLMClient, create_client
from kaprag.llm.types import (
    BackendError,
    LLMCallError,
    LLMError,
    LLMResult,
    TransientBackendError,
    Usage,
)

__all__ = [
    "BackendError",
    "LLMCallError",
    "LLMClient",
    "LLMError",
    "LLMResult",
    "TransientBackendError",
    "Usage",
    "create_client",
]
