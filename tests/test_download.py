import hashlib

import httpx
import pymupdf
import pytest

from kaprag.corpus import Document, check_document, raw_path
from kaprag.download import (
    DownloadError,
    Status,
    download_pdf,
    fetch,
    fetch_all,
    with_file_facts,
)

PDF = b"%PDF-1.7\n" + b"x" * 5000


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def doc(doc_id="akbank-2024", sha256=None) -> Document:
    bank, year = doc_id.rsplit("-", 1)
    return Document(doc_id, bank, int(year), f"https://example.test/{doc_id}.pdf", sha256, None, "")


def make_pdf(path, pages: int) -> None:
    pdf = pymupdf.open()
    for _ in range(pages):
        pdf.new_page()
    pdf.save(path)
    pdf.close()


def fake_client(routes: dict[str, httpx.Response], calls: list[str] | None = None) -> httpx.Client:
    """An httpx client served from `routes` (path -> response); unknown paths get 404."""

    def handler(request: httpx.Request) -> httpx.Response:
        if calls is not None:
            calls.append(request.url.path)
        return routes.get(request.url.path, httpx.Response(404))

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_download_pdf_writes_file_and_returns_hash(tmp_dir):
    dest = tmp_dir / "raw" / "a.pdf"
    with fake_client({"/a.pdf": httpx.Response(200, content=PDF)}) as client:
        digest = download_pdf(client, "https://example.test/a.pdf", dest, None)
    assert dest.read_bytes() == PDF
    assert digest == sha(PDF)
    assert list(dest.parent.iterdir()) == [dest]


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (httpx.Response(200, content=b"<!doctype html><html></html>"), "not a PDF"),
        (httpx.Response(200, content=b"%P"), "not a PDF"),
        (httpx.Response(403), "HTTP 403"),
    ],
)
def test_download_pdf_rejects_bad_payload_without_leaving_files(tmp_dir, response, message):
    dest = tmp_dir / "a.pdf"
    with (
        fake_client({"/a.pdf": response}) as client,
        pytest.raises(DownloadError, match=message),
    ):
        download_pdf(client, "https://example.test/a.pdf", dest, None)
    assert list(tmp_dir.iterdir()) == []


def test_download_pdf_rejects_hash_mismatch_and_keeps_existing_file(tmp_dir):
    dest = tmp_dir / "a.pdf"
    dest.write_bytes(b"%PDF old")
    with (
        fake_client({"/a.pdf": httpx.Response(200, content=PDF)}) as client,
        pytest.raises(DownloadError, match="sha256 mismatch"),
    ):
        download_pdf(client, "https://example.test/a.pdf", dest, "00" * 32)
    assert dest.read_bytes() == b"%PDF old"
    assert list(tmp_dir.iterdir()) == [dest]


def test_fetch_skips_present_file_with_matching_hash(tmp_dir):
    raw_path("akbank-2024", tmp_dir).write_bytes(PDF)
    calls: list[str] = []
    with fake_client({}, calls) as client:
        outcome = fetch(doc(sha256=sha(PDF)), tmp_dir, client)
    assert outcome.status is Status.SKIPPED
    assert calls == []


def test_fetch_redownloads_file_with_wrong_hash(tmp_dir):
    raw_path("akbank-2024", tmp_dir).write_bytes(b"%PDF corrupted")
    with fake_client({"/akbank-2024.pdf": httpx.Response(200, content=PDF)}) as client:
        outcome = fetch(doc(sha256=sha(PDF)), tmp_dir, client)
    assert outcome.status is Status.DOWNLOADED
    assert raw_path("akbank-2024", tmp_dir).read_bytes() == PDF


def test_fetch_without_hash_check_accepts_changed_content(tmp_dir):
    with fake_client({"/akbank-2024.pdf": httpx.Response(200, content=PDF)}) as client:
        outcome = fetch(doc(sha256="00" * 32), tmp_dir, client, check_sha256=False)
    assert outcome.status is Status.DOWNLOADED


def test_fetch_all_continues_after_a_failure(tmp_dir):
    docs = [doc("akbank-2023"), doc("akbank-2024"), doc("akbank-2025")]
    routes = {
        "/akbank-2023.pdf": httpx.Response(200, content=PDF),
        "/akbank-2025.pdf": httpx.Response(200, content=PDF),
    }
    with fake_client(routes) as client:
        outcomes = fetch_all(docs, tmp_dir, client)
    assert [(o.doc_id, o.status) for o in outcomes] == [
        ("akbank-2023", Status.DOWNLOADED),
        ("akbank-2024", Status.FAILED),
        ("akbank-2025", Status.DOWNLOADED),
    ]
    assert "HTTP 404" in outcomes[1].detail
    assert sorted(p.name for p in tmp_dir.iterdir()) == ["akbank-2023.pdf", "akbank-2025.pdf"]


def test_fetch_all_reports_network_errors(tmp_dir):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        (outcome,) = fetch_all([doc()], tmp_dir, client)
    assert outcome.status is Status.FAILED
    assert "ConnectError" in outcome.detail
    assert list(tmp_dir.iterdir()) == []


def test_with_file_facts_fills_hash_and_pages(tmp_dir):
    make_pdf(raw_path("akbank-2024", tmp_dir), pages=3)
    data = raw_path("akbank-2024", tmp_dir).read_bytes()

    filled, untouched = with_file_facts([doc("akbank-2024"), doc("akbank-2025")], [], tmp_dir)
    assert (filled.sha256, filled.pages) == (sha(data), 3)
    assert (untouched.sha256, untouched.pages) == (None, None)


def test_update_keeps_manifest_hash_when_download_of_corrupted_file_fails(tmp_dir):
    # Regression: a corrupted local file whose re-download fails must not have
    # its hash written to the manifest (that would make --verify pass).
    good = doc(sha256=sha(PDF))
    make_pdf(raw_path(good.doc_id, tmp_dir), pages=1)  # valid PDF, wrong content
    docs = [good]
    with fake_client({}) as client:  # every request gets 404
        outcomes = fetch_all(docs, tmp_dir, client, check_sha256=False)
    assert outcomes[0].status is Status.FAILED

    (updated,) = with_file_facts(docs, outcomes, tmp_dir)
    assert updated == good
    assert check_document(updated, tmp_dir).startswith("sha256 mismatch")
