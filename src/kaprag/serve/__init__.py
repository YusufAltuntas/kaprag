"""Serving the pipeline to people: a shared service layer (`kaprag.serve.service`), page
rendering (`kaprag.serve.pages`) and the HTTP API (`kaprag.serve.api`, needs the `app`
dependency group; not imported here)."""

from kaprag.serve.pages import PageNotFoundError, PageRenderer
from kaprag.serve.service import (
    CONFIG_ENV,
    DEFAULT_CONFIG,
    DEFAULT_ROLE,
    MAX_QUESTION_CHARS,
    AskResult,
    AskService,
    CitationView,
    InvalidRequestError,
    PipelineFailedError,
    ServiceClosedError,
    SourceView,
    TraceUsage,
    open_service,
)

__all__ = [
    "CONFIG_ENV",
    "DEFAULT_CONFIG",
    "DEFAULT_ROLE",
    "MAX_QUESTION_CHARS",
    "AskResult",
    "AskService",
    "CitationView",
    "InvalidRequestError",
    "PageNotFoundError",
    "PageRenderer",
    "PipelineFailedError",
    "ServiceClosedError",
    "SourceView",
    "TraceUsage",
    "open_service",
]
