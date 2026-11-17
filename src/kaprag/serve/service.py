"""`AskService`: one long-lived pipeline answering one question at a time, for the ask
API (`kaprag.serve.api`) and the chat demo (`scripts/chat_app.py`).

    service = open_service("router")          # loads the embedder/reranker once
    result = service.ask(question, role="analyst")
    service.close()

`open_service` builds the pipeline of `configs/<name>.yaml` exactly as the eval
harness does (same factory, same params), with an LLM client from `kaprag.llm` whose
backend is `LLM_BACKEND` (or the `backend` argument). The pipeline runs on the
service's own event loop thread, so `ask` is a plain blocking call that any thread
(a Streamlit script run, a FastAPI worker thread) may make concurrently.

Access: `role` is chosen by the caller, checked against `kaprag.access.roles.ROLES`
and passed to the pipeline as `access_role`. It is never read from the question or
from model output. The role model is simulated (see `kaprag.access.roles`).

Each request gets a `request_id` tag, merged by the pipeline into its LLM calls; the
request's API-equivalent cost and call counts are read back from the client's trace.
Each question is answered on its own: no conversation history is sent.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import tempfile
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from kaprag.access.roles import ROLES
from kaprag.corpus import BANK_NAMES
from kaprag.eval.answer import SystemAnswer
from kaprag.eval.config import CONFIG_DIR, load_config
from kaprag.eval.pipeline import PipelineContext, PipelineOutput, resolve_factory
from kaprag.llm import LLMClient, create_client
from kaprag.serve.pages import PageRenderer

log = logging.getLogger(__name__)

CONFIG_ENV = "KAPRAG_SERVE_CONFIG"
DEFAULT_CONFIG = "router"
DEFAULT_ROLE = "public"
MAX_QUESTION_CHARS = 2000
SNIPPET_CHARS = 280

UsageReader = Callable[[str], list[dict[str, Any]]]  # request id -> its LLM trace rows


class InvalidRequestError(ValueError):
    """The question or the role is not acceptable; nothing was run."""


class PipelineFailedError(RuntimeError):
    """The pipeline failed on a question. The message is short and safe to show;
    the original exception is chained and logged."""

    def __init__(self, error_type: str) -> None:
        super().__init__(f"The pipeline failed ({error_type}).")
        self.error_type = error_type


class ServiceClosedError(RuntimeError):
    """`ask` was called after `close`."""


class CitationView(BaseModel):
    doc: str
    page: int  # 1-based PDF page index
    bank: str | None  # manifest bank id; None for a doc outside the manifest
    bank_name: str | None
    year: int | None


class SourceView(BaseModel):
    """One retrieved chunk, compact: what the answer was generated from (best first)."""

    rank: int
    chunk_id: str
    doc: str
    page_start: int
    page_end: int
    score: float
    snippet: str | None


class AskResult(BaseModel):
    question: str
    role: str
    # the structured answer (kaprag.eval.answer.SystemAnswer), flattened
    answer: str
    value: int | float | None
    unit: str | None
    basis: str | None
    abstained: bool
    clarification: str | None
    citations: list[CitationView]
    sources: list[SourceView]
    route: str | None  # "simple" / "multi" when the pipeline reports it
    latency_ms: float
    cost_usd: float  # API-equivalent cost of this request's LLM calls
    llm_calls: int
    cached_llm_calls: int
    backend: str
    config: str


class TraceUsage:
    """Reads the LLM trace rows of one request from a `kaprag.llm` trace file."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def __call__(self, request_id: str) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        rows = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("tags", {}).get("request_id") == request_id:
                    rows.append(row)
        return rows


def _no_usage(request_id: str) -> list[dict[str, Any]]:
    return []


class AskService:
    def __init__(
        self,
        pipeline: Any,
        *,
        config_name: str,
        backend: str,
        pages: PageRenderer | None = None,
        usage: UsageReader | None = None,
        on_close: Callable[[], None] | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.config = config_name
        self.backend = backend
        self.pages = pages if pages is not None else PageRenderer()
        self._usage = usage or _no_usage
        self._on_close = on_close
        self._closed = False
        self._lock = threading.Lock()
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, name="kaprag-serve-loop", daemon=True
        )
        self._thread.start()

    def ask(self, question: str, role: str = DEFAULT_ROLE) -> AskResult:
        """Answer one question as `role`. Raises InvalidRequestError, PipelineFailedError
        or ServiceClosedError."""
        question = _check_question(question)
        if role not in ROLES:
            raise InvalidRequestError(f"unknown role {role!r}; expected one of {list(ROLES)}")
        if self._closed:
            raise ServiceClosedError("the service is closed")
        request_id = uuid.uuid4().hex
        tags = {"request_id": request_id, "source": "serve"}
        start = time.perf_counter()
        future = asyncio.run_coroutine_threadsafe(
            self.pipeline.run(question, tags=tags, access_role=role), self._loop
        )
        try:
            output = PipelineOutput.model_validate(future.result())
        except Exception as exc:
            log.exception("pipeline failed on request %s", request_id)
            raise PipelineFailedError(type(exc).__name__) from exc
        latency_ms = (time.perf_counter() - start) * 1000
        calls = self._usage(request_id)
        return self._result(question, role, output, latency_ms, calls)

    def _result(
        self,
        question: str,
        role: str,
        output: PipelineOutput,
        latency_ms: float,
        calls: list[dict[str, Any]],
    ) -> AskResult:
        answer: SystemAnswer = output.answer
        route = output.debug.get("route")
        return AskResult(
            question=question,
            role=role,
            answer=answer.answer,
            value=answer.value,
            unit=answer.unit,
            basis=answer.basis,
            abstained=answer.abstained,
            clarification=answer.clarification,
            citations=[self._citation(c.doc, c.page) for c in answer.citations],
            sources=[
                SourceView(
                    rank=rank,
                    chunk_id=item.chunk_id,
                    doc=item.doc,
                    page_start=item.page_start,
                    page_end=item.page_end,
                    score=item.score,
                    snippet=None if item.text is None else item.text[:SNIPPET_CHARS],
                )
                for rank, item in enumerate(output.retrieved, start=1)
            ],
            route=route if isinstance(route, str) else None,
            latency_ms=round(latency_ms, 3),
            cost_usd=round(float(sum(c.get("cost_usd") or 0 for c in calls)), 8),
            llm_calls=len(calls),
            cached_llm_calls=sum(bool(c.get("cached")) for c in calls),
            backend=self.backend,
            config=self.config,
        )

    def _citation(self, doc_id: str, page: int) -> CitationView:
        doc = self.pages.document(doc_id)
        if doc is None:
            return CitationView(doc=doc_id, page=page, bank=None, bank_name=None, year=None)
        return CitationView(
            doc=doc_id,
            page=page,
            bank=doc.bank,
            bank_name=BANK_NAMES.get(doc.bank, doc.bank),
            year=doc.year,
        )

    def page_png(self, doc_id: str, page: int) -> bytes:
        """PNG of a cited page; raises `PageNotFoundError`."""
        return self.pages.png(doc_id, page)

    def close(self) -> None:
        """Close the pipeline (models, database connection) and stop the loop; idempotent."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
        try:
            if hasattr(self.pipeline, "aclose"):
                asyncio.run_coroutine_threadsafe(self.pipeline.aclose(), self._loop).result(60)
        except Exception:
            log.exception("pipeline aclose failed")
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=30)
            if not self._thread.is_alive():
                self._loop.close()
            log.info("ask service closed (config %s)", self.config)
            if self._on_close is not None:
                self._on_close()


def _check_question(question: Any) -> str:
    if not isinstance(question, str) or not question.strip():
        raise InvalidRequestError("the question is empty")
    question = question.strip()
    if len(question) > MAX_QUESTION_CHARS:
        raise InvalidRequestError(f"the question is longer than {MAX_QUESTION_CHARS} characters")
    return question


def open_service(
    config_name: str | None = None,
    *,
    backend: str | None = None,
    config_dir: Path = CONFIG_DIR,
    llm_factory: Callable[[Path], LLMClient] | None = None,
    pages: PageRenderer | None = None,
) -> AskService:
    """Build the pipeline of `configs/<config_name>.yaml` (default: $KAPRAG_SERVE_CONFIG,
    else `router`) and keep it open. Loads the models; may take a while.

    The LLM backend is `backend`, else `LLM_BACKEND` (see `kaprag.llm.create_client`).
    `llm_factory` (trace path -> client) replaces that in tests.
    """
    name = config_name or os.environ.get(CONFIG_ENV) or DEFAULT_CONFIG
    config = load_config(name, config_dir)
    trace_dir = Path(tempfile.mkdtemp(prefix="kaprag-serve-"))
    trace_path = trace_dir / "llm-trace.jsonl"

    def cleanup() -> None:
        shutil.rmtree(trace_dir, ignore_errors=True)

    try:
        client = (
            llm_factory(trace_path)
            if llm_factory is not None
            else create_client(backend, trace_path=trace_path)
        )
        factory = resolve_factory(config.pipeline.factory)
        pipeline = factory(PipelineContext(params=dict(config.pipeline.params), llm=lambda: client))
    except BaseException:
        cleanup()
        raise
    return AskService(
        pipeline,
        config_name=name,
        backend=client.backend.name,
        pages=pages,
        usage=TraceUsage(trace_path),
        on_close=cleanup,
    )
