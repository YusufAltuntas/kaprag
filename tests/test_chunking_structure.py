import re

from kaprag.chunking.structure import chunk_structure
from kaprag.parsing.outputs import Element


def ws_spans(text: str) -> list[tuple[int, int]]:
    """Fake tokenizer: one token per whitespace-separated word."""
    return [m.span() for m in re.finditer(r"\S+", text)]


def el(
    type_: str,
    text: str,
    page: int = 1,
    path: tuple[str, ...] = (),
    *,
    pages: list[int] | None = None,
    caption: str | None = None,
    in_picture: bool = False,
) -> Element:
    return Element(
        page=page,
        pages=pages or [page],
        type=type_,
        label=type_,
        level=0 if type_ == "heading" else None,
        section_path=list(path),
        text=text,
        caption=caption,
        in_picture=in_picture,
    )


def words(prefix: str, n: int) -> str:
    return " ".join(f"{prefix}{i}" for i in range(n))


def table(n_rows: int) -> str:
    lines = ["| Kalem | 2023 |", "|---|---|"]
    lines += [f"| r{i} | {i} |" for i in range(n_rows)]
    return "\n".join(lines)


def test_running_text_is_grouped_under_its_heading_with_prefix():
    rows = [
        el("heading", "Intro", 1),
        el("paragraph", "p one", 1, ("Intro",)),
        el("list_item", "item two", 2, ("Intro",)),
        el("heading", "Risks", 3, ("Intro",)),
        el("paragraph", "r one", 3, ("Intro", "Risks")),
    ]
    chunks = chunk_structure("d-2023", rows, ws_spans, size=50)
    assert [c.text for c in chunks] == ["Intro\n\np one\nitem two", "Risks\n\nr one"]
    assert [(c.page_start, c.page_end) for c in chunks] == [(1, 2), (3, 3)]
    assert chunks[1].section_path == ("Intro", "Risks")
    assert all(c.chunk_type == "text" for c in chunks)
    assert [c.ordinal for c in chunks] == [0, 1]


def test_text_before_any_heading_has_no_prefix():
    (chunk,) = chunk_structure("d-2023", [el("paragraph", "hello world")], ws_spans, size=50)
    assert chunk.text == "hello world"
    assert chunk.section_path == ()


def test_long_group_is_split_and_every_part_carries_the_heading():
    rows = [el("paragraph", words(f"s{i}_", 4), 1, ("H",)) for i in range(5)]
    chunks = chunk_structure("d-2023", rows, ws_spans, size=10)
    assert len(chunks) > 1
    assert all(c.text.startswith("H\n\n") for c in chunks)
    assert all(c.n_tokens <= 10 for c in chunks)
    body = " ".join(c.text.removeprefix("H\n\n") for c in chunks).split()
    assert body == " ".join(r.text for r in rows).split()


def test_single_paragraph_longer_than_budget_is_split_by_tokens():
    rows = [el("paragraph", words("w", 30), 4, ("H",))]
    chunks = chunk_structure("d-2023", rows, ws_spans, size=10)
    assert all(c.n_tokens <= 10 for c in chunks)
    assert all((c.page_start, c.page_end) == (4, 4) for c in chunks)
    assert " ".join(c.text.removeprefix("H\n\n") for c in chunks).split() == words("w", 30).split()


def test_furniture_is_excluded():
    rows = [
        el("furniture", "Annual Report 2023 | 17", 1, ("H",)),
        el("paragraph", "body", 1, ("H",)),
        el("furniture", "logo", 1, ("H",), in_picture=True),
    ]
    (chunk,) = chunk_structure("d-2023", rows, ws_spans, size=50)
    assert chunk.text == "H\n\nbody"


def test_table_is_kept_whole_with_heading_and_caption_prefix():
    rows = [
        el("paragraph", "before", 5, ("Financials",)),
        el("table", table(3), 5, ("Financials",), caption="Tablo 1: Bilanço"),
        el("paragraph", "after", 6, ("Financials",)),
    ]
    chunks = chunk_structure("d-2023", rows, ws_spans, size=200)
    tables = [c for c in chunks if c.chunk_type == "table"]
    texts = [c for c in chunks if c.chunk_type == "text"]
    assert len(tables) == 1
    assert tables[0].text == "Financials\nTablo 1: Bilanço\n\n" + table(3)
    # running text is never merged with the table
    assert all("|" not in c.text for c in texts)
    assert "before" in texts[0].text and "after" in " ".join(c.text for c in texts)


def test_long_table_is_split_into_row_groups_repeating_header_rows():
    rows = [el("table", table(40), 7, ("Bilanço",), pages=[7, 8])]
    chunks = chunk_structure("d-2023", rows, ws_spans, size=40)
    assert len(chunks) > 1
    header = "| Kalem | 2023 |\n|---|---|"
    for c in chunks:
        assert c.chunk_type == "table"
        assert c.text.startswith("Bilanço\n\n" + header + "\n")
        assert c.n_tokens <= 40
        assert (c.page_start, c.page_end) == (7, 8)
    body_rows = [line for c in chunks for line in c.text.split("\n")[4:]]
    assert body_rows == [f"| r{i} | {i} |" for i in range(40)]


def test_table_row_longer_than_budget_is_split_under_the_header():
    long_row = "| " + words("x", 30) + " | 1 |"
    markdown = "| Kalem | 2023 |\n|---|---|\n| short | 2 |\n" + long_row
    chunks = chunk_structure("d-2023", [el("table", markdown, 3, ("T",))], ws_spans, size=20)
    header = "| Kalem | 2023 |\n|---|---|"
    assert all(c.text.startswith("T\n\n" + header + "\n") for c in chunks)
    assert all(c.n_tokens <= 20 for c in chunks)
    bodies = [c.text.split("\n", 4)[4] for c in chunks]
    assert bodies[0] == "| short | 2 |"
    assert " ".join(bodies[1:]).split() == long_row.split()


def test_table_header_over_half_the_budget_is_not_repeated():
    markdown = "| " + words("h", 15) + " |\n|---|\n" + "\n".join(f"| r{i} |" for i in range(20))
    chunks = chunk_structure("d-2023", [el("table", markdown, 3, ("T",))], ws_spans, size=20)
    assert all(c.n_tokens <= 20 for c in chunks)
    assert len(chunks) == 5  # 17-token header alone, then 3-token rows six at a time
    lines = [line for c in chunks for line in c.text.split("\n")[2:]]
    assert lines == markdown.split("\n")


def test_picture_text_is_separated_per_page_with_its_own_type():
    rows = [
        el("heading", "Highlights", 2),
        el("paragraph", "narrative", 2, ("Highlights",)),
        el("paragraph", "₺1,2 trilyon", 2, ("Highlights",), in_picture=True),
        el("heading", "Aktif", 2, ("Highlights",), in_picture=True),
        el("paragraph", "more narrative", 2, ("Highlights",)),
        el("paragraph", "%38 ROE", 3, ("Highlights",), in_picture=True),
    ]
    chunks = chunk_structure("d-2023", rows, ws_spans, size=50)
    text = [c for c in chunks if c.chunk_type == "text"]
    pics = [c for c in chunks if c.chunk_type == "picture_text"]
    # picture text interrupts the running text: two text chunks, same heading prefix
    assert [c.text for c in text] == ["Highlights\n\nnarrative", "Highlights\n\nmore narrative"]
    assert [c.text for c in pics] == ["Highlights\n\n₺1,2 trilyon\nAktif", "Highlights\n\n%38 ROE"]
    assert [(c.page_start, c.page_end) for c in pics] == [(2, 2), (3, 3)]
    # ordinals follow reading order of each chunk's first row
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))
    assert [c.chunk_type for c in chunks] == ["text", "picture_text", "text", "picture_text"]


def test_text_table_text_under_one_heading_gives_three_chunks_in_reading_order():
    rows = [
        el("paragraph", "before", 5, ("H",)),
        el("table", table(2), 6, ("H",)),
        el("paragraph", "after", 7, ("H",)),
    ]
    chunks = chunk_structure("d-2023", rows, ws_spans, size=200)
    assert [(c.chunk_type, c.page_start, c.page_end) for c in chunks] == [
        ("text", 5, 5),
        ("table", 6, 6),
        ("text", 7, 7),
    ]
    assert [c.text for c in (chunks[0], chunks[2])] == ["H\n\nbefore", "H\n\nafter"]
    assert [c.ordinal for c in chunks] == [0, 1, 2]


def test_fragments_between_tables_on_different_pages_do_not_merge():
    rows = [
        el("table", table(1), 10, ("H",)),
        el("paragraph", "frag a", 10, ("H",)),
        el("table", table(1), 11, ("H",)),
        el("paragraph", "frag b", 12, ("H",)),
        el("table", table(1), 13, ("H",)),
        el("paragraph", "frag c", 14, ("H",)),
    ]
    chunks = chunk_structure("d-2023", rows, ws_spans, size=200)
    texts = [c for c in chunks if c.chunk_type == "text"]
    assert [(c.text, c.page_start, c.page_end) for c in texts] == [
        ("H\n\nfrag a", 10, 10),
        ("H\n\nfrag b", 12, 12),
        ("H\n\nfrag c", 14, 14),
    ]
    assert [c.chunk_type for c in chunks] == ["table", "text"] * 3
    assert all(c.page_start == c.page_end for c in chunks)


def test_multi_page_element_spans_pages():
    rows = [el("paragraph", "spans", 9, ("H",), pages=[9, 10])]
    (chunk,) = chunk_structure("d-2023", rows, ws_spans, size=50)
    assert (chunk.page_start, chunk.page_end) == (9, 10)


def test_empty_rows_and_empty_document():
    assert chunk_structure("d-2023", [el("paragraph", "  ")], ws_spans, size=50) == []
    assert chunk_structure("d-2023", [], ws_spans, size=50) == []
