"""Rendering cited PDF pages for the ask API and the chat demo.

A page is looked up by manifest `doc_id` and 1-based PDF page index only: the file
path is built from a manifest row, never from the request, so an unknown id (including
any path-like string) is refused before the file system is touched.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from kaprag.corpus import (
    DEFAULT_MANIFEST,
    DEFAULT_RAW_DIR,
    Document,
    raw_path,
    read_manifest,
    render_page_png,
)

Reason = Literal["unknown_doc", "page_range", "missing_pdf"]


class PageNotFoundError(LookupError):
    """The requested page cannot be served; `reason` says why."""

    def __init__(self, reason: Reason, message: str) -> None:
        super().__init__(message)
        self.reason = reason


class PageRenderer:
    """Manifest-checked PNG rendering of PDF pages, with a small render cache."""

    def __init__(
        self,
        manifest_path: Path = DEFAULT_MANIFEST,
        raw_dir: Path = DEFAULT_RAW_DIR,
        *,
        zoom: float = 1.5,
        cache_size: int = 32,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.raw_dir = Path(raw_dir)
        self.zoom = zoom
        self._docs: dict[str, Document] | None = None
        self._render = lru_cache(maxsize=cache_size)(self._render_uncached)

    def _documents(self) -> dict[str, Document]:
        if self._docs is None:
            self._docs = {d.doc_id: d for d in read_manifest(self.manifest_path)}
        return self._docs

    def document(self, doc_id: str) -> Document | None:
        """The manifest row of `doc_id`, or None when it is not in the manifest."""
        return self._documents().get(doc_id)

    def png(self, doc_id: str, page: int) -> bytes:
        """PNG of one page; raises PageNotFoundError for an unknown document, a page out
        of range or a PDF that is not on disk."""
        doc = self.document(doc_id)
        if doc is None:
            raise PageNotFoundError("unknown_doc", "unknown document")
        if page < 1 or (doc.pages is not None and page > doc.pages):
            raise PageNotFoundError("page_range", "page out of range")
        return self._render(doc.doc_id, page)

    def _render_uncached(self, doc_id: str, page: int) -> bytes:
        try:
            return render_page_png(raw_path(doc_id, self.raw_dir), page, self.zoom)
        except FileNotFoundError:
            raise PageNotFoundError("missing_pdf", "PDF not available locally") from None
        except IndexError:
            raise PageNotFoundError("page_range", "page out of range") from None
