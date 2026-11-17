"""RAG context: numbered sources with bank / year / doc / page labels."""

from kaprag.eval.pipeline import RetrievedItem
from kaprag.rag.context import format_context, source_label


def item(doc, start, end, text, rank=1):
    return RetrievedItem(
        chunk_id=f"{doc}:fixed:{rank}",
        doc=doc,
        page_start=start,
        page_end=end,
        score=1.0 - rank / 100,
        text=text,
    )


def test_source_label_single_page_and_span():
    assert source_label(1, item("akbank-2024", 87, 87, "x")) == (
        "[1] Banka: Akbank | Yıl: 2024 | Doküman: akbank-2024 | Sayfa: 87"
    )
    assert source_label(3, item("isbank-2023", 10, 12, "x")) == (
        "[3] Banka: İş Bankası | Yıl: 2023 | Doküman: isbank-2023 | Sayfa: 10-12"
    )


def test_source_label_unknown_bank_falls_back_to_id():
    assert source_label(2, item("ziraat-2025", 5, 5, "x")).startswith(
        "[2] Banka: ziraat | Yıl: 2025 |"
    )


def test_format_context_numbers_sources_in_rank_order():
    items = [
        item("akbank-2024", 87, 87, "Net kâr 42,1 milyar TL.", 1),
        item("garanti-2025", 50, 51, "SYO %18,2.", 2),
    ]
    text = format_context(items)
    assert text == (
        "[1] Banka: Akbank | Yıl: 2024 | Doküman: akbank-2024 | Sayfa: 87\n"
        "Net kâr 42,1 milyar TL.\n"
        "\n"
        "[2] Banka: Garanti BBVA | Yıl: 2025 | Doküman: garanti-2025 | Sayfa: 50-51\n"
        "SYO %18,2."
    )


def test_format_context_strips_chunk_whitespace_and_handles_missing_text():
    items = [item("yapikredi-2023", 3, 3, "  metin \n"), item("akbank-2025", 4, 4, None, 2)]
    text = format_context(items)
    assert "Sayfa: 3\nmetin\n\n[2]" in text
    assert text.endswith("Sayfa: 4\n")


def test_format_context_empty():
    assert format_context([]) == ""
