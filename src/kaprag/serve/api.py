"""The ask API (FastAPI) over an `AskService`. Needs the `app` dependency group.

    POST /ask                    {"question": str, "role": "public"|"analyst"|"risk"}
    GET  /health
    GET  /page/{doc_id}/{page}   PNG of a cited PDF page (1-based page index)

Launch with `scripts/serve_api.py`. There is no authentication: this is a local demo,
bound to 127.0.0.1 unless the launcher is told otherwise explicitly.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Literal

import anyio
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field, field_validator

from kaprag.access.roles import ROLES
from kaprag.serve.pages import PageNotFoundError
from kaprag.serve.service import (
    MAX_QUESTION_CHARS,
    AskResult,
    AskService,
    InvalidRequestError,
    PipelineFailedError,
    ServiceClosedError,
)

DESCRIPTION = (
    "Citation-giving question answering over the 2023-2025 annual reports of four Turkish "
    "banks. Questions may be in Turkish or English (answers usually, but not always, follow "
    "the question's language); citations are (doc, page) with a "
    "1-based PDF page index, viewable at /page/{doc}/{page}.\n\n"
    "No authentication: this is a local demo, bound to 127.0.0.1 by default.\n\n"
    "Access roles (public, analyst, risk) are SIMULATED: the reports are public documents "
    "and their chunks carry roles assigned by rule only to demonstrate access filtering. "
    "The role is taken only from the request's `role` field, never from the question."
)

Role = Literal[ROLES]  # type: ignore[valid-type]  # the one role vocabulary


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    role: Role = "public"

    @field_validator("question")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("the question is empty")
        return value


class ErrorResponse(BaseModel):
    error: str
    detail: str


class HealthResponse(BaseModel):
    status: str
    config: str
    backend: str
    roles: list[str]


def _error(status: int, error: str, detail: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": error, "detail": detail})


def create_app(service: AskService, *, close_on_shutdown: bool = False) -> FastAPI:
    """The API over an open service. With `close_on_shutdown`, the app closes the service
    when the server shuts down (the launcher's case); otherwise the caller closes it."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        yield
        if close_on_shutdown:
            await anyio.to_thread.run_sync(service.close)

    app = FastAPI(title="kaprag", description=DESCRIPTION, version="0.1.0", lifespan=lifespan)

    @app.exception_handler(Exception)
    async def _unexpected(request: Request, exc: Exception) -> JSONResponse:
        return _error(500, "internal_error", "Unexpected server error.")

    @app.post(
        "/ask",
        response_model=AskResult,
        responses={500: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    )
    def ask(request: AskRequest) -> AskResult | JSONResponse:
        # A sync endpoint: FastAPI runs it in a worker thread; `ask` blocks on the
        # service's own event loop.
        try:
            return service.ask(request.question, role=request.role)
        except InvalidRequestError as exc:
            return _error(422, "invalid_request", str(exc))
        except PipelineFailedError as exc:
            return _error(500, "pipeline_error", str(exc))
        except ServiceClosedError:
            return _error(503, "unavailable", "The service is shutting down.")

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok", config=service.config, backend=service.backend, roles=list(ROLES)
        )

    @app.get(
        "/page/{doc_id}/{page}",
        response_class=Response,
        responses={200: {"content": {"image/png": {}}}, 404: {"model": ErrorResponse}},
    )
    def page(doc_id: str, page: int) -> Response:
        try:
            png = service.page_png(doc_id, page)
        except PageNotFoundError as exc:
            return _error(404, exc.reason, str(exc))
        return Response(content=png, media_type="image/png")

    return app
