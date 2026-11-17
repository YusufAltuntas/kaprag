import json

import pytest

from kaprag.parsing.outputs import (
    Element,
    PageText,
    iter_elements,
    iter_pages,
    parsed_path,
    write_jsonl_atomic,
)

ELEMENT_ROW = {
    "page": 3,
    "pages": [3, 4],
    "type": "table",
    "label": "table",
    "level": None,
    "section_path": ["BİLANÇO", "Varlıklar"],
    "text": "| a | b |\n|---|---|\n| 1 | 2 |",
    "caption": "Tablo 1",
    "in_picture": False,
}


def test_parsed_path_layout(tmp_dir):
    assert (
        parsed_path("docling", "akbank-2024", tmp_dir) == tmp_dir / "docling" / "akbank-2024.jsonl"
    )
    with pytest.raises(ValueError, match="unknown parser"):
        parsed_path("pdfplumber", "akbank-2024", tmp_dir)


def test_write_jsonl_atomic_writes_utf8_and_leaves_no_temp_file(tmp_dir):
    path = tmp_dir / "sub" / "x.jsonl"
    assert write_jsonl_atomic(path, iter([{"page": 1, "text": "Şirket"}])) == 1
    assert path.read_text(encoding="utf-8") == '{"page": 1, "text": "Şirket"}\n'
    assert [p.name for p in path.parent.iterdir()] == ["x.jsonl"]


def test_iter_pages_and_elements_round_trip(tmp_dir):
    pages = tmp_dir / "p.jsonl"
    write_jsonl_atomic(pages, [{"page": 1, "text": "a"}, {"page": 2, "text": ""}])
    assert list(iter_pages(pages)) == [PageText(1, "a"), PageText(2, "")]

    elements = tmp_dir / "e.jsonl"
    write_jsonl_atomic(elements, [ELEMENT_ROW])
    assert list(iter_elements(elements)) == [Element(**ELEMENT_ROW)]


@pytest.mark.parametrize(
    "row",
    [
        {"page": 1},
        {"page": 1, "text": "a", "extra": 1},
    ],
)
def test_iter_pages_rejects_unexpected_keys(tmp_dir, row):
    path = tmp_dir / "p.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="row keys"):
        list(iter_pages(path))


def test_iter_elements_rejects_missing_section_path(tmp_dir):
    row = {k: v for k, v in ELEMENT_ROW.items() if k != "section_path"}
    path = tmp_dir / "e.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="row keys"):
        list(iter_elements(path))


def test_iter_jsonl_reports_bad_line(tmp_dir):
    path = tmp_dir / "p.jsonl"
    path.write_text('{"page": 1, "text": "a"}\n{broken\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"p\.jsonl:2"):
        list(iter_pages(path))


def test_iter_elements_rejects_rows_without_in_picture(tmp_dir):
    row = {k: v for k, v in ELEMENT_ROW.items() if k != "in_picture"}
    path = tmp_dir / "e.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="row keys"):
        list(iter_elements(path))
