import json
import logging

import pytest

from kaprag.parsing.batch_runner import (
    Job,
    final_path,
    is_complete,
    marker_path,
    page_ranges,
    part_path,
    run_jobs,
)
from kaprag.parsing.outputs import iter_elements


def test_page_ranges():
    assert page_ranges(1, 7, 3) == [(1, 3), (4, 6), (7, 7)]
    assert page_ranges(5, 5, 10) == [(5, 5)]
    with pytest.raises(ValueError):
        page_ranges(1, 2, 0)


class FakeConverter:
    """Returns one heading and one paragraph per page; fails on the given pages."""

    def __init__(self, fail_pages=()):
        self.fail_pages = set(fail_pages)
        self.calls: list[tuple[str, int, int]] = []

    def __call__(self, pdf_path, first, last):
        self.calls.append((pdf_path.stem, first, last))
        if self.fail_pages & set(range(first, last + 1)):
            raise RuntimeError("CUDA out of memory")
        rows = []
        for page in range(first, last + 1):
            rows.append(_row(page, "heading", f"H{page}", level=1))
            rows.append(_row(page, "paragraph", f"text {page}"))
        return rows


def _row(page, type_, text, level=None):
    return {
        "page": page,
        "pages": [page],
        "type": type_,
        "label": "section_header" if type_ == "heading" else "text",
        "level": level,
        "text": text,
        "caption": None,
    }


def make_job(tmp_dir, doc_id="akbank-2024", first=1, last=5) -> Job:
    pdf = tmp_dir / "raw" / f"{doc_id}.pdf"
    pdf.parent.mkdir(exist_ok=True)
    pdf.write_bytes(b"%PDF-1.7\n")
    return Job(doc_id, pdf, first, last)


def test_run_jobs_writes_final_rows_with_section_paths_and_marker(tmp_dir):
    out = tmp_dir / "out"
    job = make_job(tmp_dir)
    summary = run_jobs([job], out, FakeConverter(), pages_per_batch=2)
    assert summary.completed == ["akbank-2024"]
    assert summary.failures == []
    assert summary.pages_converted == 5
    elements = list(iter_elements(final_path(out, job.doc_id)))
    assert [e.page for e in elements] == [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    assert elements[3].section_path == ["H2"]
    assert elements[2].section_path == []  # a level-1 heading closes the previous one
    meta = json.loads(marker_path(out, job.doc_id).read_text(encoding="utf-8"))
    assert meta == {"doc_id": "akbank-2024", "first_page": 1, "last_page": 5, "rows": 10}
    assert sorted(p.name for p in out.iterdir()) == ["akbank-2024.done.json", "akbank-2024.jsonl"]


def test_run_jobs_isolates_a_failing_range_and_resumes_only_missing_work(tmp_dir, caplog):
    out = tmp_dir / "out"
    bad = make_job(tmp_dir, "akbank-2024", 1, 6)
    good = make_job(tmp_dir, "isbank-2024", 1, 2)
    first_run = FakeConverter(fail_pages={3})
    released = []
    with caplog.at_level(logging.INFO, logger="kaprag.parsing.batch_runner"):
        summary = run_jobs(
            [bad, good], out, first_run, pages_per_batch=2, on_failure=lambda: released.append(1)
        )
    assert [(f.doc_id, f.first, f.last) for f in summary.failures] == [("akbank-2024", 3, 4)]
    assert "CUDA out of memory" in summary.failures[0].error
    assert released == [1]
    assert summary.completed == ["isbank-2024"]
    assert first_run.calls == [
        ("akbank-2024", 1, 2),
        ("akbank-2024", 3, 4),
        ("akbank-2024", 5, 6),
        ("isbank-2024", 1, 2),
    ]
    # The failed document has parts but no final file or marker.
    assert not final_path(out, bad.doc_id).exists()
    assert not is_complete(out, bad)
    assert part_path(out, bad.doc_id, 1, 2).is_file()
    assert not part_path(out, bad.doc_id, 3, 4).exists()
    assert "akbank-2024 pages 3-4 FAILED" in caplog.text
    assert "pages/s" in caplog.text

    second_run = FakeConverter()
    summary = run_jobs([bad, good], out, second_run, pages_per_batch=2)
    assert second_run.calls == [("akbank-2024", 3, 4)]
    assert summary.skipped == ["isbank-2024"]
    assert summary.completed == ["akbank-2024"]
    assert [e.page for e in iter_elements(final_path(out, bad.doc_id))] == [
        p for p in range(1, 7) for _ in range(2)
    ]
    assert not (out / "akbank-2024.parts").exists()


def test_a_page_subset_is_not_taken_for_the_whole_document(tmp_dir):
    out = tmp_dir / "out"
    sample = make_job(tmp_dir, first=2, last=3)
    run_jobs([sample], out, FakeConverter(), pages_per_batch=10)
    assert is_complete(out, sample)
    whole = make_job(tmp_dir, first=1, last=5)
    assert not is_complete(out, whole)
    converter = FakeConverter()
    run_jobs([whole], out, converter, pages_per_batch=10)
    assert converter.calls == [("akbank-2024", 1, 5)]
    assert is_complete(out, whole)


def test_final_file_without_marker_is_redone(tmp_dir):
    out = tmp_dir / "out"
    job = make_job(tmp_dir, last=2)
    out.mkdir()
    final_path(out, job.doc_id).write_text("partial garbage\n", encoding="utf-8")
    converter = FakeConverter()
    summary = run_jobs([job], out, converter, pages_per_batch=10)
    assert converter.calls == [("akbank-2024", 1, 2)]
    assert summary.completed == ["akbank-2024"]
    assert len(list(iter_elements(final_path(out, job.doc_id)))) == 4


def test_missing_pdf_is_a_document_failure(tmp_dir):
    job = make_job(tmp_dir)
    job.pdf_path.unlink()
    converter = FakeConverter()
    summary = run_jobs([job], tmp_dir / "out", converter, pages_per_batch=2)
    assert converter.calls == []
    assert [(f.doc_id, f.first) for f in summary.failures] == [("akbank-2024", None)]
