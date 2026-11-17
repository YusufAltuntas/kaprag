"""Structure-aware parser: Docling layout analysis with table structure recognition.

Output rows follow the Docling schema in `kaprag.parsing.outputs`. How they are built:

- A document is converted in bounded page ranges (see `kaprag.parsing.batch_runner`);
  each range yields "part rows" (all fields except `section_path`), and
  `assign_section_paths` adds the heading hierarchy once the parts are joined, so
  the hierarchy runs across range boundaries.
- Elements come from `DoclingDocument.iterate_items` over the body and furniture
  layers, in Docling's reading order, descending into pictures
  (`traverse_pictures=True`). Page headers/footers are kept as type `furniture`
  (labelled by Docling, not stripped here). Pictures themselves carry no text and
  get no row; their captions are kept as `caption` rows.
- Text Docling finds inside a picture region (infographic figures, KPI boxes, chart
  labels) is kept as rows with `in_picture: true`, right after the picture's
  position in reading order. Such rows never open or close a section: a heading
  inside a picture does not change `section_path`.
- Docling joins a line ending in "-" to the next line without the hyphen
  (de-hyphenation). Between two numbers that turns a range wrapped over a line
  ("2013-" / "2022") into one wrong number ("20132022"); `keep_numeric_line_hyphens`
  keeps such hyphens, giving "2013- 2022".
- List items carry their marker (e.g. "1.") in front of the text, as in the PDF.
- A table becomes one row: its Markdown export (column padding removed, see
  `compact_markdown_table`) as `text`; captions Docling linked
  to it go to `caption` and are not repeated as separate rows.
- Headings: `title` -> level 0, `section_header` -> Docling's level. Levels are
  inferred by Docling's heading-hierarchy step (PDF bookmarks, numbering, font
  style) within one converted page range, so levels and `section_path` are
  approximate (see the limitation in `kaprag.parsing.outputs`).
- Page provenance: `pages` lists every page in the element's provenance, `page` is
  the first. Docling does not merge elements across the page ranges we convert, so
  an element never spans two ranges.
- OCR is off (born-digital PDFs with a text layer); table structure is on
  (TableFormer, accurate mode, cell matching against the PDF text cells).

Docling is imported lazily so the package works without the `ml` dependency group.
"""

import ctypes
import importlib.util
import io
import logging
import re
import sys
from collections.abc import Iterable
from pathlib import Path

log = logging.getLogger(__name__)

TABLE_LABELS = {"table", "document_index"}
_TYPE_BY_LABEL = {
    "title": "heading",
    "section_header": "heading",
    "text": "paragraph",
    "paragraph": "paragraph",
    "list_item": "list_item",
    "table": "table",
    "document_index": "table",
    "caption": "caption",
    "page_header": "furniture",
    "page_footer": "furniture",
}
_SKIPPED_LABELS = {"picture", "chart"}


def compact_markdown_table(markdown: str) -> str:
    """Drop the column-alignment padding of Docling's Markdown tables.

    Docling pads every cell to its column's width, which roughly triples the size of
    wide financial tables; the table stays valid Markdown without it.
    """
    lines = []
    for line in markdown.splitlines():
        line = re.sub(r" {2,}", " ", line)
        if re.fullmatch(r"\|(?:\s*:?-+:?\s*\|)+", line):
            line = re.sub(r"-{3,}", "---", line)
        lines.append(line)
    return "\n".join(lines)


def keep_numeric_line_hyphens(lines: list[str]) -> list[str]:
    """Protect a line-final hyphen between two numbers from Docling's de-hyphenation.

    A space after the hyphen makes Docling join the lines as usual but keep the "-".
    """
    out = list(lines)
    for ix in range(len(out) - 1):
        line, nxt = out[ix], out[ix + 1].lstrip()
        if line.endswith("-") and line[-2:-1].isdigit() and nxt[:1].isdigit():
            out[ix] = line + " "
    return out


def _patch_docling_dehyphenation() -> None:
    """Make Docling's text assembly apply `keep_numeric_line_hyphens` (idempotent)."""
    from docling.models.stages.page_assemble.page_assemble_model import PageAssembleModel

    original = PageAssembleModel.sanitize_text
    if getattr(original, "_kaprag_patched", False):
        return

    def sanitize_text(self, lines):
        return original(self, keep_numeric_line_hyphens(lines))

    sanitize_text._kaprag_patched = True
    PageAssembleModel.sanitize_text = sanitize_text


def _label(item) -> str:
    label = item.label
    return str(getattr(label, "value", label))


def items_to_part_rows(items: Iterable[tuple], doc) -> list[dict]:
    """Map Docling items (in reading order) to part rows (rows without `section_path`).

    `items` are `(item, depth)` pairs as yielded by `DoclingDocument.iterate_items`
    with `traverse_pictures=True`; `doc` is the document they belong to (needed for
    table Markdown and captions). An item is inside a picture if it follows the
    picture at a greater depth and is not one of the picture's captions.
    """
    items = list(items)
    table_caption_refs = {
        ref.cref for item, _ in items if _label(item) in TABLE_LABELS for ref in item.captions
    }
    rows: list[dict] = []
    last_page = None
    open_pictures: list[tuple[int, set[str]]] = []  # (depth, caption refs) of enclosing pictures
    for item, depth in items:
        while open_pictures and depth <= open_pictures[-1][0]:
            open_pictures.pop()
        label = _label(item)
        in_picture = any(item.self_ref not in captions for _, captions in open_pictures)
        if label in _SKIPPED_LABELS:
            open_pictures.append((depth, {ref.cref for ref in item.captions}))
            continue
        if item.self_ref in table_caption_refs:
            continue
        caption = None
        if label in TABLE_LABELS:
            text = compact_markdown_table(item.export_to_markdown(doc=doc))
            caption = item.caption_text(doc) or None
        else:
            text = getattr(item, "text", "") or ""
            marker = (getattr(item, "marker", "") or "") if label == "list_item" else ""
            if marker.strip() and text.strip():
                text = f"{marker.strip()} {text}"
        if not text.strip():
            continue
        pages = sorted({prov.page_no for prov in item.prov})
        if not pages:
            if last_page is None:
                continue
            pages = [last_page]
        last_page = pages[-1]
        level = None
        if label == "title":
            level = 0
        elif label == "section_header":
            level = int(item.level)
        rows.append(
            {
                "page": pages[0],
                "pages": pages,
                "type": _TYPE_BY_LABEL.get(label, "other"),
                "label": label,
                "level": level,
                "text": text,
                "caption": caption,
                "in_picture": in_picture,
            }
        )
    return rows


def document_part_rows(doc, content_layers: set) -> list[dict]:
    """Part rows of a converted DoclingDocument, including text inside pictures."""
    items = doc.iterate_items(included_content_layers=content_layers, traverse_pictures=True)
    return items_to_part_rows(items, doc)


def assign_section_paths(rows: Iterable[dict]) -> Iterable[dict]:
    """Add `section_path` to part rows of a whole document, in reading order.

    A heading closes every open heading of the same or a deeper level. A heading's
    own path holds its ancestors; any other row's path holds all open headings.
    Headings inside pictures (`in_picture`) are treated like any other row.
    """
    stack: list[tuple[int, str]] = []
    for row in rows:
        if row["type"] == "heading" and not row["in_picture"]:
            while stack and stack[-1][0] >= row["level"]:
                stack.pop()
            path = [text for _, text in stack]
            stack.append((row["level"], row["text"]))
        else:
            path = [text for _, text in stack]
        yield {**row, "section_path": path}


def _import_docling_parse_from_ascii_path() -> None:
    """Work around docling-parse failing to load its resources from a non-ASCII path.

    On Windows the docling-parse extension opens its bundled resource files with a
    narrow-character API, which fails when site-packages sits under a non-ASCII
    directory (e.g. a user name with "ş"). Importing the package through the 8.3
    short form of that directory gives it an ASCII path. No-op elsewhere.
    """
    if sys.platform != "win32" or "docling_parse" in sys.modules:
        return
    spec = importlib.util.find_spec("docling_parse")
    if spec is None or spec.origin is None:
        return
    site_dir = str(Path(spec.origin).parent.parent)
    if site_dir.isascii():
        return
    buf = ctypes.create_unicode_buffer(32768)
    if not ctypes.windll.kernel32.GetShortPathNameW(site_dir, buf, len(buf)) or not (
        buf.value.isascii()
    ):
        raise RuntimeError(
            f"docling-parse cannot load its resources from the non-ASCII path {site_dir!r} "
            "and no ASCII 8.3 short path is available; move the virtualenv to an ASCII path"
        )
    sys.path.insert(0, buf.value)
    try:
        import docling_parse  # noqa: F401
    finally:
        sys.path.remove(buf.value)


class DoclingConverter:
    """Converts page ranges of a PDF to part rows with one loaded Docling pipeline."""

    def __init__(self, device: str = "cuda", model_batch_size: int = 4, num_threads: int = 4):
        _import_docling_parse_from_ascii_path()
        from docling.datamodel.accelerator_options import AcceleratorOptions
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import (
            PdfPipelineOptions,
            TableFormerMode,
            TableStructureOptions,
        )
        from docling.document_converter import DocumentConverter, PdfFormatOption

        options = PdfPipelineOptions(
            do_ocr=False,
            do_table_structure=True,
            table_structure_options=TableStructureOptions(
                mode=TableFormerMode.ACCURATE, do_cell_matching=True
            ),
            layout_batch_size=model_batch_size,
            table_batch_size=model_batch_size,
            accelerator_options=AcceleratorOptions(device=device, num_threads=num_threads),
        )
        options.heading_hierarchy_options.enabled = True
        _patch_docling_dehyphenation()
        self.device = device
        self._converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
        )
        self._converter.initialize_pipeline(InputFormat.PDF)

    def __call__(self, pdf_path: Path, first: int, last: int) -> list[dict]:
        """Part rows for PDF pages `first`..`last` (1-based, inclusive).

        Raises if Docling does not report full success for the range.
        """
        from docling.datamodel.base_models import ConversionStatus
        from docling_core.types.doc import ContentLayer
        from docling_core.types.io import DocumentStream

        # Passed as a stream: the native PDF parser cannot open non-ASCII paths on Windows.
        stream = DocumentStream(name=pdf_path.name, stream=io.BytesIO(pdf_path.read_bytes()))
        result = self._converter.convert(stream, page_range=(first, last), raises_on_error=False)
        if result.status is not ConversionStatus.SUCCESS:
            errors = "; ".join(e.error_message for e in result.errors) or "no error details"
            raise RuntimeError(f"Docling status {result.status.value}: {errors}")
        return document_part_rows(result.document, {ContentLayer.BODY, ContentLayer.FURNITURE})

    def peak_memory_gb(self) -> float | None:
        """Peak CUDA memory allocated by this process so far, or None off the GPU."""
        import torch

        if not torch.cuda.is_available():
            return None
        return torch.cuda.max_memory_allocated() / 1e9

    def release_memory(self) -> None:
        """Return cached CUDA memory after a failed range (e.g. out of memory)."""
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
