"""Real Docling on a tiny generated PDF (needs the `ml` group, the models and a CUDA GPU)."""

import pymupdf
import pytest

from kaprag.parsing.batch_runner import Job, final_path, run_jobs
from kaprag.parsing.outputs import iter_elements

pytestmark = [
    pytest.mark.gpu,
    # Raised inside Docling for its own default options, not by our settings.
    pytest.mark.filterwarnings("ignore:`force_full_page_ocr` is deprecated:DeprecationWarning"),
    pytest.mark.filterwarnings(
        "ignore:This field is deprecated. Use `generate_page_images:DeprecationWarning"
    ),
]

ROWS = [
    ("Kalem", "2024", "2023"),
    ("Krediler", "1.234.567", "987.654"),
    ("Mevduat", "2.345", "1.111"),
]


def make_report_pdf(path) -> None:
    pdf = pymupdf.open()
    for page_no in (1, 2):
        page = pdf.new_page()
        page.insert_text((72, 80), f"Finansal Tablolar {page_no}", fontsize=20)
        page.insert_text((72, 120), "Tutarlar bin TL olarak ifade edilmistir.", fontsize=10)
        x0, y0, w, h = 72, 160, 150, 24
        for r, row in enumerate(ROWS):
            for c, cell in enumerate(row):
                rect = pymupdf.Rect(x0 + c * w, y0 + r * h, x0 + (c + 1) * w, y0 + (r + 1) * h)
                page.draw_rect(rect, color=(0, 0, 0), width=0.5)
                page.insert_text((rect.x0 + 4, rect.y1 - 7), cell, fontsize=10)
    pdf.save(path)
    pdf.close()


def test_docling_converts_a_generated_pdf_with_page_provenance(tmp_dir):
    from kaprag.parsing.docling_parser import DoclingConverter

    pdf = tmp_dir / "ş" / "rapor-2024.pdf"  # non-ASCII path, like the repo's
    pdf.parent.mkdir()
    make_report_pdf(pdf)
    out = tmp_dir / "out"
    summary = run_jobs(
        [Job("rapor-2024", pdf, 1, 2)], out, DoclingConverter(device="cuda"), pages_per_batch=1
    )
    assert summary.failures == []
    elements = list(iter_elements(final_path(out, "rapor-2024")))
    assert {e.page for e in elements} == {1, 2}
    text = "\n".join(e.text for e in elements)
    assert "Finansal Tablolar 1" in text and "Finansal Tablolar 2" in text
    tables = [e for e in elements if e.type == "table"]
    assert tables, [(e.type, e.text) for e in elements]
    assert "1.234.567" in tables[0].text and "|" in tables[0].text
