"""Fakes for the ask service, the ask API and the chat demo tests: a scripted pipeline,
a tiny corpus (manifest + one generated PDF) and a fake LLM backend."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from kaprag.eval.answer import Citation, SystemAnswer
from kaprag.eval.pipeline import PipelineContext, PipelineOutput, RetrievedItem
from kaprag.llm.types import BackendResponse, Usage

FAKE_ROOT_ENV = "KAPRAG_FAKE_SERVE_ROOT"
MANIFEST_HEADER = "doc_id,bank,year,url,sha256,pages,source_note\n"


def write_corpus(root: Path) -> tuple[Path, Path]:
    """`root/manifest.csv` (akbank-2024 with a 3-page PDF on disk, garanti-2023 listed
    but not downloaded) and `root/raw/`. Returns (manifest, raw_dir)."""
    import pymupdf

    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    with pymupdf.open() as pdf:
        for n in range(1, 4):
            pdf.new_page(width=200, height=200).insert_text((20, 40), f"page {n}")
        pdf.save(raw / "akbank-2024.pdf")
    manifest = root / "manifest.csv"
    manifest.write_text(
        MANIFEST_HEADER
        + "akbank-2024,akbank,2024,https://example.invalid/a.pdf,abc,3,\n"
        + "garanti-2023,garanti,2023,https://example.invalid/g.pdf,,,\n",
        encoding="utf-8",
    )
    return manifest, raw


def item(doc: str, page: int, score: float, text: str = "metin") -> RetrievedItem:
    return RetrievedItem(
        chunk_id=f"{doc}-{page}", doc=doc, page_start=page, page_end=page, score=score, text=text
    )


def answer_output(route: str | None = "simple") -> PipelineOutput:
    debug: dict[str, Any] = {} if route is None else {"route": route}
    return PipelineOutput(
        answer=SystemAnswer(
            answer="Akbank'ın 2024 sonunda 694 şubesi vardı [1].",
            value=694,
            unit="adet",
            basis=None,
            citations=[Citation(doc="akbank-2024", page=2), Citation(doc="garanti-2023", page=7)],
        ),
        retrieved=[item("akbank-2024", 2, 0.9, "x" * 1000), item("garanti-2023", 7, 0.5)],
        debug=debug,
    )


class FakePipeline:
    """Records every call; returns `output` or raises `error`."""

    def __init__(self, output: Any = None, error: Exception | None = None) -> None:
        self.output = output if output is not None else answer_output()
        self.error = error
        self.calls: list[dict[str, Any]] = []
        self.closed = 0

    async def run(
        self, question: str, *, tags: dict[str, Any], access_role: str | None = None
    ) -> Any:
        self.calls.append({"question": question, "tags": dict(tags), "access_role": access_role})
        if self.error is not None:
            raise self.error
        return self.output

    async def aclose(self) -> None:
        self.closed += 1


class ScriptedPipeline(FakePipeline):
    """Answers by keyword: "yok" abstains, "hangi" asks for clarification, "hata" fails,
    anything else answers with citations."""

    async def run(
        self, question: str, *, tags: dict[str, Any], access_role: str | None = None
    ) -> Any:
        self.calls.append({"question": question, "tags": dict(tags), "access_role": access_role})
        if "hata" in question:
            raise RuntimeError("backend exploded with a long internal message")
        if "yok" in question:
            return PipelineOutput(
                answer=SystemAnswer(answer="Bu bilgi belgelerde yok.", abstained=True),
                retrieved=[item("akbank-2024", 1, 0.2)],
                debug={"route": "simple"},
            )
        if "hangi" in question:
            return PipelineOutput(
                answer=SystemAnswer(answer="", clarification="Hangi bankayı soruyorsunuz?"),
                retrieved=[],
                debug={"route": "simple"},
            )
        return answer_output(route="multi")


class FakeBackend:
    name = "fake_backend"

    async def generate(self, system: str, user: str, model: str) -> BackendResponse:
        return BackendResponse(text="ok", usage=Usage(input_tokens=1000, output_tokens=100))


class LLMCallingPipeline(FakePipeline):
    """Makes one LLM call per question through the context's client, tags merged."""

    def __init__(self, ctx: PipelineContext) -> None:
        super().__init__()
        self.ctx = ctx

    async def run(
        self, question: str, *, tags: dict[str, Any], access_role: str | None = None
    ) -> Any:
        await self.ctx.llm().complete("s", question, "claude-haiku-5-5", tags={**tags, "step": "x"})
        return await super().run(question, tags=tags, access_role=access_role)


def build_llm_calling_pipeline(ctx: PipelineContext) -> LLMCallingPipeline:
    """A harness factory, referenced by dotted path from a test config."""
    return LLMCallingPipeline(ctx)


# The chat demo test points KAPRAG_SERVE_FACTORY here.
LAST_SERVICE: list[Any] = []


def open_fake_service(config_name: str | None = None, **_: Any) -> Any:
    """An AskService over a ScriptedPipeline and the corpus under $KAPRAG_FAKE_SERVE_ROOT."""
    from kaprag.serve import AskService, PageRenderer

    manifest, raw = write_corpus(Path(os.environ[FAKE_ROOT_ENV]))
    service = AskService(
        ScriptedPipeline(),
        config_name=config_name or "fake",
        backend="claude_code",
        pages=PageRenderer(manifest, raw),
    )
    LAST_SERVICE.append(service)
    return service
