"""Page features, strata, skips, rendering and seeded sampling."""

from decimal import Decimal

import pytest
from gold_helpers import elements, synthetic_doc, write_parsed

from kaprag.gold.pages import DocPages, PageInfo, page_info, render_page, sample_pages


def infos(year=2024):
    rows = synthetic_doc(year)
    return {p: page_info("akbank-2024", year, p, elements(rows, p)) for p in range(1, 7)}


def test_strata_and_skips():
    i = infos()
    assert i[1].skip_reason == "cover"
    assert i[2].skip_reason == "toc"
    assert i[5].skip_reason == "near_empty"
    assert i[3].stratum == "table" and i[3].has_comparative
    assert i[4].stratum == "text" and not i[4].has_comparative
    assert i[6].stratum == "mixed"


def _info(page: int, stratum: str) -> PageInfo:
    return PageInfo("akbank-2024", page, stratum, None, 1000, 0, 0, 0, False)


def test_sampling_is_seeded_per_document_and_stratum():
    pool = [_info(p, "table" if p % 2 else "text") for p in range(1, 101)]
    a = sample_pages(pool, 4, 3)
    assert a == sample_pages(list(reversed(pool)), 4, 3)
    assert [i.page for i in a] == sorted(i.page for i in a)
    assert sum(i.stratum == "table" for i in a) == 4 and sum(i.stratum == "text" for i in a) == 3
    assert a != sample_pages(pool, 4, 3, seed=7)
    assert len(sample_pages(pool[:3], 10, 10)) == 3  # small strata give all they have


def test_sampling_one_document_at_a_time():
    other = PageInfo("isbank-2024", 1, "table", None, 1000, 0, 0, 0, False)
    with pytest.raises(ValueError):
        sample_pages([_info(1, "table"), other], 1, 1)


def test_render_drops_furniture_and_marks_tables():
    text = render_page(elements(synthetic_doc(), 3))
    assert "Sayfa 3" not in text
    assert "## KONSOLİDE FİNANSAL TABLOLAR 2024" in text
    assert "[Tablo]\n| | Cari Dönem" in text
    assert len(render_page(elements(synthetic_doc(), 3), max_chars=50)) < 120


def test_doc_pages_numbers_from_both_parsers(tmp_dir):
    parsed = write_parsed(tmp_dir, "akbank-2024")
    pages = DocPages.load("akbank-2024", parsed)
    numbers = pages.numbers(3)
    assert Decimal("45210300") in numbers
    assert Decimal("-210500000") in numbers and Decimal("210500000") in numbers
    assert Decimal("17.30") in numbers
    assert [i.stratum for i in pages.infos(2024)][:4] == ["skip", "skip", "table", "text"]
