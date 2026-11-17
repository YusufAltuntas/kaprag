"""Docling item -> row mapping, with stand-ins for Docling items (no `ml` group needed)."""

from dataclasses import dataclass, field

from kaprag.parsing.docling_parser import (
    assign_section_paths,
    compact_markdown_table,
    document_part_rows,
    items_to_part_rows,
    keep_numeric_line_hyphens,
)


@dataclass
class Prov:
    page_no: int


@dataclass
class Ref:
    cref: str

    def resolve(self, doc):
        return doc.by_ref[self.cref]


@dataclass
class Item:
    self_ref: str
    label: str
    pages: list[int] = field(default_factory=list)
    text: str = ""
    level: int | None = None
    captions: list[Ref] = field(default_factory=list)
    marker: str = ""
    parent: Ref | None = None

    @property
    def prov(self):
        return [Prov(p) for p in self.pages]


@dataclass
class Table(Item):
    markdown: str = ""
    caption: str = ""

    def export_to_markdown(self, doc):
        return self.markdown

    def caption_text(self, doc):
        return self.caption


GROUP_LABELS = {"group", "list"}


class FakeDoc:
    """Stand-in for a DoclingDocument tree, iterated the way docling-core 2.107 does it.

    `tree` is a list of `(item, [child nodes])`. Parent refs are set from the tree.
    `iterate_items` yields `(item, depth)` with depth = the length of the traversal
    stack, which also counts group levels although groups themselves are not
    yielded; picture children other than captions are only visited with
    `traverse_pictures=True`.
    """

    def __init__(self, tree):
        self.tree = tree
        self.by_ref = {}

        def register(nodes, parent):
            for item, children in nodes:
                item.parent = Ref(parent.self_ref) if parent is not None else None
                self.by_ref[item.self_ref] = item
                register(children, item)

        register(tree, None)

    def iterate_items(self, included_content_layers=None, traverse_pictures=False):
        def walk(nodes, depth):
            for item, children in nodes:
                if item.label not in GROUP_LABELS:
                    yield item, depth
                captions = {ref.cref for ref in item.captions}
                for child in children:
                    hidden = item.label == "picture" and not traverse_pictures
                    if hidden and child[0].self_ref not in captions:
                        continue
                    yield from walk([child], depth + 1)

        return walk(self.tree, 1)


def rows_of(tree):
    doc = FakeDoc(tree)
    return items_to_part_rows(doc.iterate_items(traverse_pictures=True), doc)


def test_items_to_part_rows_maps_types_levels_tables_and_pages():
    rows = rows_of(
        [
            (Item("#/texts/0", "page_header", [5], "2024 Entegre Faaliyet Raporu"), []),
            (Item("#/texts/1", "title", [5], "AKBANK T.A.Ş."), []),
            (Item("#/texts/2", "section_header", [5], "BİLANÇO", level=2), []),
            (
                Item("#/pictures/0", "picture", [5], captions=[Ref("#/texts/3")]),
                [(Item("#/texts/3", "caption", [5], "Logo"), [])],
            ),
            (
                Table(
                    "#/tables/0",
                    "table",
                    [5, 6],
                    captions=[Ref("#/texts/4")],
                    markdown="| VARLIKLAR | 2024 |\n|---|---|\n| Nakit | 1.234 |",
                    caption="I. Bilanço",
                ),
                [(Item("#/texts/4", "caption", [5], "I. Bilanço"), [])],
            ),
            (
                Item("#/groups/0", "list"),
                [(Item("#/texts/5", "list_item", [6], "Bulunmamaktadır.", marker="1."), [])],
            ),
            (Item("#/texts/6", "text", [6], "   "), []),
            (Item("#/texts/7", "footnote", [6], "(*) Dipnot"), []),
            (Item("#/texts/8", "text", [], "no provenance"), []),
        ]
    )
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
    assert rows[5]["text"] == "1. Bulunmamaktadır."
    assert not any(r["in_picture"] for r in rows)  # a picture caption is not picture text
    assert all("section_path" not in r for r in rows)


def test_items_to_part_rows_keeps_text_inside_pictures():
    rows = rows_of(
        [
            (Item("#/texts/0", "section_header", [27], "Finansal Göstergeler", level=1), []),
            (
                Item("#/pictures/0", "picture", [27], captions=[Ref("#/texts/4")]),
                [
                    (Item("#/texts/1", "text", [27], "Aktif Büyüklük"), []),
                    (Item("#/texts/2", "section_header", [27], "Kârlılık", level=1), []),
                    (
                        Item("#/pictures/1", "chart", [27]),
                        [(Item("#/texts/3", "text", [27], "%31,2"), [])],
                    ),
                    (Item("#/texts/4", "caption", [27], "Grafik 1"), []),
                    (
                        Item("#/groups/0", "group"),
                        [(Item("#/texts/5", "text", [28], "1.234.567 TL"), [])],
                    ),
                ],
            ),
            (Item("#/texts/6", "text", [28], "After the picture."), []),
        ]
    )
    assert [(r["text"], r["page"], r["in_picture"]) for r in rows] == [
        ("Finansal Göstergeler", 27, False),
        ("Aktif Büyüklük", 27, True),
        ("Kârlılık", 27, True),
        ("%31,2", 27, True),
        ("Grafik 1", 27, False),
        ("1.234.567 TL", 28, True),
        ("After the picture.", 28, False),
    ]
    # A heading inside a picture does not open a section.
    paths = [r["section_path"] for r in assign_section_paths(rows)]
    assert paths == [[]] + [["Finansal Göstergeler"]] * 6


def test_grouped_items_after_a_picture_are_not_picture_text():
    """Regression: a list group right after a picture has the same depth as the
    picture's children, but its items are not inside the picture."""
    rows = rows_of(
        [
            (Item("#/texts/0", "section_header", [14], "Öne Çıkanlar", level=1), []),
            (
                Item("#/pictures/0", "picture", [14]),
                [(Item("#/texts/1", "text", [14], "52,8 milyar TL"), [])],
            ),
            (
                Item("#/groups/0", "list"),
                [
                    (Item("#/texts/2", "list_item", [14], "first bullet after the picture"), []),
                    (Item("#/texts/3", "section_header", [14], "Grouped heading", level=2), []),
                ],
            ),
            (Item("#/texts/4", "text", [14], "Running text."), []),
        ]
    )
    assert [(r["text"], r["in_picture"]) for r in rows] == [
        ("Öne Çıkanlar", False),
        ("52,8 milyar TL", True),
        ("first bullet after the picture", False),
        ("Grouped heading", False),
        ("Running text.", False),
    ]
    paths = [r["section_path"] for r in assign_section_paths(rows)]
    assert paths[4] == ["Öne Çıkanlar", "Grouped heading"]


def test_items_to_part_rows_table_without_caption():
    rows = rows_of([(Table("#/tables/0", "document_index", [2], markdown="| a |"), [])])
    assert rows == [
        {
            "page": 2,
            "pages": [2],
            "type": "table",
            "label": "document_index",
            "level": None,
            "text": "| a |",
            "caption": None,
            "in_picture": False,
        }
    ]


def row(type_, text, level=None):
    return {"type": type_, "text": text, "level": level, "in_picture": False}


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


def test_document_part_rows_descends_into_pictures():
    doc = FakeDoc(
        [
            (
                Item("#/pictures/0", "picture", [14], captions=[Ref("#/texts/2")]),
                [
                    (Item("#/texts/0", "text", [14], "Net Kâr"), []),
                    (Item("#/texts/1", "text", [14], "52,8 milyar TL"), []),
                    (Item("#/texts/2", "caption", [14], "Öne çıkanlar"), []),
                ],
            ),
            (Item("#/texts/3", "text", [14], "Running text."), []),
        ]
    )
    rows = document_part_rows(doc, {"body", "furniture"})
    assert [(r["text"], r["in_picture"]) for r in rows] == [
        ("Net Kâr", True),
        ("52,8 milyar TL", True),
        ("Öne çıkanlar", False),
        ("Running text.", False),
    ]


def test_keep_numeric_line_hyphens_only_between_numbers():
    lines = [
        "Departmanında, 1999-",
        "2002 Küçük",
        "algo-",
        "rithms",
        "sayfa 174-",
        " 195;",
        "- madde",
    ]
    assert keep_numeric_line_hyphens(lines) == [
        "Departmanında, 1999- ",
        "2002 Küçük",
        "algo-",
        "rithms",
        "sayfa 174- ",
        " 195;",
        "- madde",
    ]
