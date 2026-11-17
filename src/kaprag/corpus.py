"""The document corpus: manifest rows, local PDF paths and file checks.

`data/manifest.csv` is the single list of documents. Later stages (parsing,
indexing, the gold set, the review app) read documents through this module
instead of parsing the CSV themselves.
"""

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "data" / "manifest.csv"
DEFAULT_RAW_DIR = REPO_ROOT / "data" / "raw"

MANIFEST_COLUMNS = ("doc_id", "bank", "year", "url", "sha256", "pages", "source_note")


class ManifestError(ValueError):
    """The manifest file is malformed."""


@dataclass(frozen=True)
class Document:
    """One manifest row. `sha256` and `pages` are None until the file is downloaded."""

    doc_id: str
    bank: str
    year: int
    url: str
    sha256: str | None
    pages: int | None
    source_note: str


def read_manifest(path: Path = DEFAULT_MANIFEST) -> list[Document]:
    """Read and validate the manifest, in file order."""
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if tuple(reader.fieldnames or ()) != MANIFEST_COLUMNS:
            raise ManifestError(
                f"{path}: expected columns {list(MANIFEST_COLUMNS)}, got {reader.fieldnames}"
            )
        docs = [_parse_row(row, line=reader.line_num, path=path) for row in reader]
    seen: set[str] = set()
    for doc in docs:
        if doc.doc_id in seen:
            raise ManifestError(f"{path}: duplicate doc_id {doc.doc_id!r}")
        seen.add(doc.doc_id)
    return docs


def _parse_row(row: dict[str, str], line: int, path: Path) -> Document:
    try:
        year = int(row["year"])
        pages = int(row["pages"]) if row["pages"] else None
    except ValueError as e:
        raise ManifestError(f"{path}:{line}: {e}") from e
    doc = Document(
        doc_id=row["doc_id"],
        bank=row["bank"],
        year=year,
        url=row["url"],
        sha256=row["sha256"].lower() or None,
        pages=pages,
        source_note=row["source_note"],
    )
    if doc.doc_id != f"{doc.bank}-{doc.year}":
        raise ManifestError(f"{path}:{line}: doc_id {doc.doc_id!r} is not '<bank>-<year>'")
    if not doc.url:
        raise ManifestError(f"{path}:{line}: empty url for {doc.doc_id}")
    return doc


def write_manifest(docs: list[Document], path: Path = DEFAULT_MANIFEST) -> None:
    """Write the manifest (UTF-8, LF line endings), replacing the file atomically."""
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(MANIFEST_COLUMNS)
        for d in docs:
            pages = "" if d.pages is None else d.pages
            writer.writerow(
                [d.doc_id, d.bank, d.year, d.url, d.sha256 or "", pages, d.source_note]
            )
    tmp.replace(path)


def raw_path(doc_id: str, raw_dir: Path = DEFAULT_RAW_DIR) -> Path:
    """Local path of a document's PDF: `data/raw/<doc_id>.pdf`."""
    return raw_dir / f"{doc_id}.pdf"


def sha256_file(path: Path) -> str:
    """Hex sha256 of a file, read in chunks."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def pdf_page_count(path: Path) -> int:
    """Number of PDF pages (not printed page numbers)."""
    import pymupdf  # imported lazily: only needed when filling the manifest

    with pymupdf.open(path) as pdf:
        return pdf.page_count


def check_document(doc: Document, raw_dir: Path = DEFAULT_RAW_DIR) -> str | None:
    """Offline check of a document's local file; returns an error message or None if it is ok."""
    if doc.sha256 is None:
        return "not downloaded (no sha256 in manifest)"
    path = raw_path(doc.doc_id, raw_dir)
    if not path.is_file():
        return f"file missing: {path}"
    actual = sha256_file(path)
    if actual != doc.sha256:
        return f"sha256 mismatch: manifest {doc.sha256}, file {actual}"
    return None
