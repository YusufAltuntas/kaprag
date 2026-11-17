"""Download the manifest's PDFs into `data/raw/`, idempotently and atomically."""

import hashlib
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path

import httpx

from kaprag.corpus import Document, pdf_page_count, raw_path, sha256_file

PDF_MAGIC = b"%PDF"
# Generic browser-like User-Agent; no personal or project information.
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
)
TIMEOUT = httpx.Timeout(30.0, read=120.0)


class DownloadError(Exception):
    """A document could not be downloaded or failed a payload check."""


class Status(StrEnum):
    SKIPPED = "skipped"
    DOWNLOADED = "downloaded"
    FAILED = "failed"


@dataclass(frozen=True)
class Outcome:
    doc_id: str
    status: Status
    detail: str = ""


def make_client() -> httpx.Client:
    return httpx.Client(
        headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT, follow_redirects=True
    )


def download_pdf(client: httpx.Client, url: str, dest: Path, expected_sha256: str | None) -> str:
    """Stream `url` to `dest` via a temp file; returns the payload's sha256.

    The payload must start with `%PDF` and, if given, match `expected_sha256`;
    otherwise `dest` is left untouched and DownloadError is raised.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    h = hashlib.sha256()
    head = b""
    try:
        with client.stream("GET", url) as resp, tmp.open("wb") as f:
            if resp.status_code != 200:
                raise DownloadError(f"HTTP {resp.status_code} for {url}")
            for chunk in resp.iter_bytes():
                if len(head) < len(PDF_MAGIC):
                    head += chunk[: len(PDF_MAGIC)]
                    if len(head) >= len(PDF_MAGIC) and not head.startswith(PDF_MAGIC):
                        raise DownloadError(f"payload is not a PDF (starts with {head[:8]!r})")
                h.update(chunk)
                f.write(chunk)
        if not head.startswith(PDF_MAGIC):
            raise DownloadError(f"payload is not a PDF (starts with {head!r})")
        digest = h.hexdigest()
        if expected_sha256 is not None and digest != expected_sha256:
            raise DownloadError(f"sha256 mismatch: manifest {expected_sha256}, download {digest}")
        tmp.replace(dest)
        return digest
    except httpx.HTTPError as e:
        raise DownloadError(f"{type(e).__name__}: {e}") from e
    finally:
        tmp.unlink(missing_ok=True)


def fetch(doc: Document, raw_dir: Path, client: httpx.Client, check_sha256: bool = True) -> Outcome:
    """Download one document unless a file with the expected content is already present.

    A present file is kept when the manifest has no sha256 yet or the hash matches.
    With `check_sha256=False` a fresh download is accepted whatever its hash
    (used when refreshing the manifest).
    """
    dest = raw_path(doc.doc_id, raw_dir)
    if dest.is_file() and (doc.sha256 is None or sha256_file(dest) == doc.sha256):
        return Outcome(doc.doc_id, Status.SKIPPED, "already present")
    try:
        download_pdf(client, doc.url, dest, doc.sha256 if check_sha256 else None)
    except DownloadError as e:
        return Outcome(doc.doc_id, Status.FAILED, str(e))
    return Outcome(doc.doc_id, Status.DOWNLOADED, f"{dest.stat().st_size} bytes")


def fetch_all(
    docs: list[Document], raw_dir: Path, client: httpx.Client, check_sha256: bool = True
) -> list[Outcome]:
    """Fetch every document; a failure is recorded and does not stop the others."""
    return [fetch(doc, raw_dir, client, check_sha256) for doc in docs]


def with_file_facts(
    docs: list[Document], outcomes: list[Outcome], raw_dir: Path
) -> list[Document]:
    """Fill `sha256` and `pages` from the local files.

    Rows without a file, and rows whose fetch failed (a stale or corrupted file
    may still be on disk), keep their manifest values.
    """
    failed = {o.doc_id for o in outcomes if o.status is Status.FAILED}
    out = []
    for doc in docs:
        path = raw_path(doc.doc_id, raw_dir)
        if doc.doc_id not in failed and path.is_file():
            doc = replace(doc, sha256=sha256_file(path), pages=pdf_page_count(path))
        out.append(doc)
    return out
