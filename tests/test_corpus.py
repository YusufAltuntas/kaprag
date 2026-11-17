import hashlib

import pytest

from kaprag.corpus import (
    DEFAULT_MANIFEST,
    MANIFEST_COLUMNS,
    Document,
    ManifestError,
    check_document,
    raw_path,
    read_manifest,
    render_page_png,
    sha256_file,
    write_manifest,
)

HEADER = ",".join(MANIFEST_COLUMNS)


def doc(doc_id="akbank-2024", sha256=None, pages=None) -> Document:
    bank, year = doc_id.rsplit("-", 1)
    url = f"https://example.test/{doc_id}.pdf"
    return Document(doc_id, bank, int(year), url, sha256, pages, "n")


def test_read_manifest_parses_typed_rows(tmp_dir):
    path = tmp_dir / "manifest.csv"
    path.write_text(
        f"{HEADER}\n"
        "isbank-2024,isbank,2024,https://x.test/Y%C4%B1ll%C4%B1k.pdf,ABCDEF,299,"
        '"İş Bankası, 2-up"\n'
        "akbank-2025,akbank,2025,https://x.test/a.pdf,,,not yet downloaded\n",
        encoding="utf-8",
    )
    docs = read_manifest(path)
    assert docs == [
        Document(
            "isbank-2024", "isbank", 2024, "https://x.test/Y%C4%B1ll%C4%B1k.pdf",
            "abcdef", 299, "İş Bankası, 2-up",
        ),
        Document("akbank-2025", "akbank", 2025, "https://x.test/a.pdf", None, None,
                 "not yet downloaded"),
    ]


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("doc_id,bank,year\n", "expected columns"),
        (f"{HEADER}\nakbank-2024,akbank,2023,u,,,n\n", "<bank>-<year>"),
        (f"{HEADER}\nakbank-2024,akbank,2024,u,,x,n\n", "invalid literal"),
        (f"{HEADER}\nakbank-2024,akbank,2024,,,,n\n", "empty url"),
        (f"{HEADER}\nakbank-2024,akbank,2024,u,,,n\nakbank-2024,akbank,2024,u,,,n\n", "duplicate"),
    ],
)
def test_read_manifest_rejects_malformed(tmp_dir, content, message):
    path = tmp_dir / "manifest.csv"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ManifestError, match=message):
        read_manifest(path)


def test_write_then_read_round_trips(tmp_dir):
    path = tmp_dir / "manifest.csv"
    docs = [doc("garanti-2023", "ab" * 32, 172), doc("yapikredi-2025")]
    write_manifest(docs, path)
    assert read_manifest(path) == docs
    assert not (tmp_dir / "manifest.csv.tmp").exists()


def test_repo_manifest_is_valid():
    docs = read_manifest(DEFAULT_MANIFEST)
    assert {d.bank for d in docs} == {"garanti", "akbank", "isbank", "yapikredi"}
    assert {d.year for d in docs} == {2023, 2024, 2025}
    assert len(docs) == 12


def test_raw_path_is_doc_id_pdf(tmp_dir):
    assert raw_path("akbank-2024", tmp_dir) == tmp_dir / "akbank-2024.pdf"


def test_sha256_file(tmp_dir):
    path = tmp_dir / "f.bin"
    path.write_bytes(b"%PDF-1.7 data")
    assert sha256_file(path) == hashlib.sha256(b"%PDF-1.7 data").hexdigest()


def test_check_document_passes_on_matching_hash(tmp_dir):
    raw_path("akbank-2024", tmp_dir).write_bytes(b"%PDF-1.7 data")
    d = doc(sha256=hashlib.sha256(b"%PDF-1.7 data").hexdigest())
    assert check_document(d, tmp_dir) is None


def test_check_document_fails_on_mismatch_missing_and_undownloaded(tmp_dir):
    raw_path("akbank-2024", tmp_dir).write_bytes(b"%PDF-1.7 changed")
    assert "sha256 mismatch" in check_document(doc(sha256="00" * 32), tmp_dir)
    assert "file missing" in check_document(doc("akbank-2023", sha256="00" * 32), tmp_dir)
    assert "not downloaded" in check_document(doc(), tmp_dir)


def test_render_page_png(tmp_dir):
    import pymupdf

    path = tmp_dir / "two.pdf"
    with pymupdf.open() as pdf:
        pdf.new_page(width=200, height=100)
        pdf.new_page(width=400, height=100)
        pdf.save(path)
    png = render_page_png(path, 2, zoom=1.0)
    assert png[1:4] == b"PNG"
    assert pymupdf.Pixmap(png).width == 400
    assert pymupdf.Pixmap(render_page_png(path, 2, zoom=2.0)).width == 800
    with pytest.raises(IndexError):
        render_page_png(path, 3)
    with pytest.raises(IndexError):
        render_page_png(path, 0)
    with pytest.raises(FileNotFoundError):
        render_page_png(tmp_dir / "missing.pdf", 1)
