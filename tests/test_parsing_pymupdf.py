import pymupdf

from kaprag.parsing.outputs import iter_jsonl, iter_pages
from kaprag.parsing.pymupdf_parser import extract_pages, parse_document


def make_pdf(path, texts: list[str]) -> None:
    pdf = pymupdf.open()
    for text in texts:
        page = pdf.new_page()
        if text:
            page.insert_text((72, 72), text)
    pdf.save(path)
    pdf.close()


def test_extract_pages_one_row_per_page_in_order(tmp_dir):
    pdf = tmp_dir / "doc.pdf"
    make_pdf(pdf, ["Birinci sayfa 12.778", "", "Third page"])
    pages = extract_pages(pdf)
    assert [p.page for p in pages] == [1, 2, 3]
    assert "Birinci sayfa 12.778" in pages[0].text
    assert pages[1].text == ""
    assert "Third page" in pages[2].text


def test_parse_document_writes_exact_schema_and_is_idempotent(tmp_dir):
    pdf = tmp_dir / "doc.pdf"
    make_pdf(pdf, ["Akbank", "İş Bankası"])
    out = tmp_dir / "out" / "doc.jsonl"
    assert parse_document(pdf, out) == 2
    first = out.read_bytes()
    rows = list(iter_jsonl(out))
    assert [set(r) for r in rows] == [{"page", "text"}] * 2
    assert [p.page for p in iter_pages(out)] == [1, 2]
    assert parse_document(pdf, out) == 2
    assert out.read_bytes() == first
    assert sorted(p.name for p in out.parent.iterdir()) == ["doc.jsonl"]
