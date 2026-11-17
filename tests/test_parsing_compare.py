from kaprag.corpus import Document
from kaprag.parsing.compare import (
    compare_page,
    default_choices,
    numeric_tokens,
    pick_table_page,
)
from kaprag.parsing.outputs import Element, write_jsonl_atomic


def el(page, type_="paragraph", text="x", pages=None, in_picture=False):
    return Element(page, pages or [page], type_, type_, None, [], text, None, in_picture)


def as_row(e: Element) -> dict:
    return dict(e.__dict__)


def doc(doc_id):
    bank, year = doc_id.rsplit("-", 1)
    return Document(doc_id, bank, int(year), "https://example.test/x.pdf", None, 3, "")


def test_numeric_tokens_keep_turkish_separators():
    assert numeric_tokens("Toplam 1.592.375.704 TL, oran %12,5 (2024)") == [
        "1.592.375.704",
        "12,5",
        "2024",
    ]


def test_pick_table_page_prefers_most_tables_then_lowest_page():
    elements = [el(1), el(2, "table", "| a |"), el(3, "table", "| b |"), el(3, "table", "| c |")]
    assert pick_table_page(elements) == 3
    assert pick_table_page([el(4, "table", "|x|"), el(2, "table", "|y|")]) == 2
    assert pick_table_page([el(1)]) is None


def test_default_choices_one_document_per_bank_with_tables(tmp_dir):
    write_jsonl_atomic(tmp_dir / "akbank-2023.jsonl", [as_row(el(1))])
    write_jsonl_atomic(tmp_dir / "akbank-2024.jsonl", [as_row(el(2, "table", "|t|"))])
    write_jsonl_atomic(tmp_dir / "akbank-2025.jsonl", [as_row(el(3, "table", "|t|"))])
    write_jsonl_atomic(tmp_dir / "isbank-2024.jsonl", [as_row(el(1, "table", "|t|"))])
    docs = [
        doc(d) for d in ("akbank-2023", "akbank-2024", "akbank-2025", "garanti-2024", "isbank-2024")
    ]
    assert default_choices(docs, tmp_dir) == [("akbank-2024", 2), ("isbank-2024", 1)]


def test_compare_page_renders_facts_and_both_outputs(tmp_dir):
    mu, dl = tmp_dir / "pymupdf", tmp_dir / "docling"
    write_jsonl_atomic(
        mu / "akbank-2024.jsonl",
        [{"page": 1, "text": "kapak"}, {"page": 2, "text": "Nakit 1.234\n5.678\n9"}],
    )
    write_jsonl_atomic(
        dl / "akbank-2024.jsonl",
        [
            as_row(el(1, text="kapak")),
            as_row(el(2, "table", "| Nakit | 1.234 | 5.678 |")),
            as_row(el(2, "furniture", "2024 Raporu")),
        ],
    )
    out = compare_page("akbank-2024", 2, mu, dl)
    assert "akbank-2024  PDF page 2" in out
    assert "1 tables" in out and "2 elements" in out
    assert "PyMuPDF :     19 chars,     3 numeric tokens" in out
    assert (
        "numeric tokens in PyMuPDF but not in Docling: 1; in Docling but not in PyMuPDF: 1" in out
    )
    assert "| Nakit | 1.234 | 5.678 |" in out
    assert "kapak" not in out


def test_compare_page_marks_text_inside_pictures(tmp_dir):
    mu, dl = tmp_dir / "pymupdf", tmp_dir / "docling"
    write_jsonl_atomic(mu / "garanti-2023.jsonl", [{"page": 1, "text": "Net kâr 52,8"}])
    write_jsonl_atomic(
        dl / "garanti-2023.jsonl", [as_row(el(1, text="Net kâr 52,8", in_picture=True))]
    )
    out = compare_page("garanti-2023", 1, mu, dl)
    assert "[paragraph p1 in-picture]" in out
    assert "numeric tokens in PyMuPDF but not in Docling: 0" in out
