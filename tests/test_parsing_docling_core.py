"""Docling row mapping on real docling-core documents (needs the `ml` group, no GPU)."""

from importlib.util import find_spec

import pytest

from kaprag.parsing.docling_parser import document_part_rows

pytestmark = pytest.mark.skipif(
    find_spec("docling_core") is None, reason="docling-core is in the ml dependency group"
)


def test_list_after_a_picture_is_not_picture_text():
    from docling_core.types.doc import (
        BoundingBox,
        ContentLayer,
        DocItemLabel,
        DoclingDocument,
        ProvenanceItem,
    )

    def prov(text):
        return ProvenanceItem(
            page_no=14, bbox=BoundingBox(l=0, t=0, r=10, b=10), charspan=(0, len(text))
        )

    doc = DoclingDocument(name="garanti-2023")
    doc.add_heading("Öne Çıkanlar", prov=prov("Öne Çıkanlar"))
    picture = doc.add_picture(prov=prov(""))
    doc.add_text(DocItemLabel.TEXT, "52,8 milyar TL", prov=prov("52,8 milyar TL"), parent=picture)
    group = doc.add_list_group()
    bullet = "first bullet after the picture"
    doc.add_list_item(bullet, marker="•", prov=prov(bullet), parent=group)
    doc.add_text(DocItemLabel.TEXT, "Running text.", prov=prov("Running text."))

    rows = document_part_rows(doc, {ContentLayer.BODY, ContentLayer.FURNITURE})
    assert [(r["text"], r["in_picture"], r["page"]) for r in rows] == [
        ("Öne Çıkanlar", False, 14),
        ("52,8 milyar TL", True, 14),
        ("• first bullet after the picture", False, 14),
        ("Running text.", False, 14),
    ]
