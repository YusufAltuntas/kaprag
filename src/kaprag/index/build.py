"""Chunking a document for a variant from its parsed output, and chunk statistics."""

import statistics
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from kaprag.chunking.base import Chunk, TokenSpans
from kaprag.chunking.fixed import chunk_fixed
from kaprag.chunking.structure import chunk_structure
from kaprag.corpus import Document, raw_path
from kaprag.parsing.batch_runner import Job, is_complete
from kaprag.parsing.outputs import DEFAULT_PARSED_DIR, iter_elements, iter_pages, parsed_path

# Which parser's output each chunking variant reads.
VARIANT_PARSER = {"fixed": "pymupdf", "structure": "docling"}


class MissingInput(Exception):
    """The document's parsed input for a variant does not exist (yet)."""


def chunk_document(
    variant: str,
    doc: Document,
    token_spans: TokenSpans,
    parsed_dir: Path = DEFAULT_PARSED_DIR,
) -> list[Chunk]:
    """Chunks of one document for a variant, read from its parser's output.

    Raises MissingInput when that output is missing; for Docling, the document must
    also have its completion marker covering every page.
    """
    parser = VARIANT_PARSER[variant]
    path = parsed_path(parser, doc.doc_id, parsed_dir)
    if not path.is_file():
        raise MissingInput(f"{doc.doc_id}: no {parser} output at {path}")
    if variant == "fixed":
        return chunk_fixed(doc.doc_id, iter_pages(path), token_spans)
    if doc.pages is None:
        raise MissingInput(f"{doc.doc_id}: page count missing from the manifest")
    job = Job(doc.doc_id, raw_path(doc.doc_id), 1, doc.pages)
    if not is_complete(path.parent, job):
        raise MissingInput(f"{doc.doc_id}: docling output incomplete (no completion marker)")
    return chunk_structure(doc.doc_id, iter_elements(path), token_spans)


@dataclass(frozen=True)
class ChunkStats:
    count: int
    tokens_min: int
    tokens_median: float
    tokens_max: int
    by_type: dict[str, int]

    def describe(self) -> str:
        types = ", ".join(
            f"{t} {n} ({n / self.count:.1%})" for t, n in sorted(self.by_type.items())
        )
        return (
            f"{self.count} chunks, tokens min/median/max "
            f"{self.tokens_min}/{self.tokens_median:g}/{self.tokens_max}; {types}"
        )


def chunk_stats(chunks: Sequence[Chunk]) -> ChunkStats:
    tokens = [c.n_tokens for c in chunks] or [0]
    return ChunkStats(
        count=len(chunks),
        tokens_min=min(tokens),
        tokens_median=statistics.median(tokens),
        tokens_max=max(tokens),
        by_type=dict(Counter(c.chunk_type for c in chunks)),
    )
