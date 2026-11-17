"""Docling item -> row mapping, with stand-ins for Docling items (no `ml` group needed)."""

from dataclasses import dataclass, field

from kaprag.parsing.docling_parser import (
    assign_section_paths,
    compact_markdown_table,
    items_to_part_rows,
)


@dataclass
class Prov:
    page_no: int


@dataclass
class Ref:
    cref: str


@dataclass
class Item:
    self_ref: str
    label: str
    pages: list[int]
    text: str = ""
    level: int | None = None
    captions: list[Ref] = field(default_factory=list)

    @property
    def prov(self):
        return [Prov(p) for p in self.pages]


@dataclass
class Table(Item):
    markdown: str = ""
    caption: str = ""

    def export_to_markdown(self, doc):
        assert doc is DOC
        return self.markdown

    def caption_text(self, doc):
        assert doc is DOC
        return self.caption


DOC = object()


def test_items_to_part_rows_maps_types_levels_tables_and_pages():
    items = [
        Item("#/texts/0", "page_header", [5], "2024 Entegre Faaliyet Raporu"),
        Item("#/texts/1", "title", [5], "AKBANK T.A.Ş."),
        Item("#/texts/2", "section_header", [5], "BİLANÇO", level=2),
        Item("#/pictures/0", "picture", [5]),
        Item("#/texts/3", "caption", [5], "Logo"),
        Table(
            "#/tables/0",
            "table",
            [5, 6],
            captions=[Ref("#/texts/4")],
            markdown="| VARLIKLAR | 2024 |\n|---|---|\n| Nakit | 1.234 |",
            caption="I. Bilanço",
        ),
        Item("#/texts/4", "caption", [5], "I. Bilanço"),
        Item("#/texts/5", "list_item", [6], "Bulunmamaktadır."),
        Item("#/texts/6", "text", [6], "   "),
        Item("#/texts/7", "footnote", [6], "(*) Dipnot"),
        Item("#/texts/8", "text", [], "no provenance"),
    ]
    rows = items_to_part_rows(items, DOC)
    assert [(r["type"], r["label"], r["level"], r["page"], r["pages"]) for r in rows] == [
        ("furniture", "page_header", None, 5, [5]),
        ("heading", "title", 0, 5, [5]),
        ("heading", "section_header", 2, 5, [5]),
        ("caption", "caption", None, 5, [5]),
        ("table", "table", None, 5, [5, 6]),
        ("list_item", "list_item", None, 6, [6]),
        ("other", "footnote", None, 6, [6]),
        ("paragraph", "text", None, 6, [6]),
    ]
    table = rows[4]
    assert table["text"].startswith("| VARLIKLAR")
    assert table["caption"] == "I. Bilanço"
    assert all(r["caption"] is None for r in rows if r is not table)
    assert all("section_path" not in r for r in rows)


def test_items_to_part_rows_table_without_caption():
    rows = items_to_part_rows([Table("#/tables/0", "document_index", [2], markdown="| a |")], DOC)
    assert rows == [
        {
            "page": 2,
            "pages": [2],
            "type": "table",
            "label": "document_index",
            "level": None,
            "text": "| a |",
            "caption": None,
        }
    ]


def row(type_, text, level=None):
    return {"type": type_, "text": text, "level": level}


def test_assign_section_paths_nests_and_closes_headings():
    rows = [
        row("paragraph", "before any heading"),
        row("heading", "Doc", 0),
        row("heading", "A", 1),
        row("heading", "A.1", 2),
        row("paragraph", "in A.1"),
        row("heading", "A.1.x", 4),
        row("table", "t"),
        row("heading", "B", 1),
        row("paragraph", "in B"),
        row("heading", "B'", 1),
    ]
    paths = [r["section_path"] for r in assign_section_paths(rows)]
    assert paths == [
        [],
        [],
        ["Doc"],
        ["Doc", "A"],
        ["Doc", "A", "A.1"],
        ["Doc", "A", "A.1"],
        ["Doc", "A", "A.1", "A.1.x"],
        ["Doc"],
        ["Doc", "B"],
        ["Doc"],
    ]


def test_compact_markdown_table_drops_padding():
    padded = (
        "| VARLIKLAR      | CARİ DÖNEM - TP   |\n"
        "|----------------|-------------------|\n"
        "| Nakit          | 287.113.554       |"
    )
    assert compact_markdown_table(padded) == (
        "| VARLIKLAR | CARİ DÖNEM - TP |\n|---|---|\n| Nakit | 287.113.554 |"
    )
