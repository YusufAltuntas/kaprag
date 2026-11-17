"""Page drafts: figure normalisation against the page, wording checks, and the check stage."""

from decimal import Decimal

import pytest
from gold_helpers import write_parsed

from kaprag.gold.checking import CheckItem, PageCheck, apply_check, build_prompt
from kaprag.gold.drafting import DraftOut, PageDrafts, PageTask, validate_draft, validate_page
from kaprag.gold.figures import FigureError, read_figure
from kaprag.gold.pages import DocPages
from kaprag.gold.text import bases_mentioned, copied_run

TASK = PageTask(
    "akbank-2024", "akbank", 2024, 3, {"tablo": 2, "hesaplama": 1, "olgusal": 1, "sentez": 1}
)


@pytest.fixture
def page(tmp_dir):
    pages = DocPages.load("akbank-2024", write_parsed(tmp_dir, "akbank-2024"))
    return pages.render(3), pages.numbers(3)


def out(**kw) -> DraftOut:
    base = dict(
        type="tablo",
        question="Akbank 2024 yılında konsolide bazda ne kadar net kâr elde etti?",
        answer="45.210.300 bin TL",
        year=2024,
        basis="konsolide",
        figure="45.210.300",
        unit_text="bin TL",
        value=45_210_300_000,
        unit="TL",
        prior_figure=None,
        prior_value=None,
        key_points=[],
    )
    return DraftOut(**(base | kw))


def test_read_figure_applies_the_unit_multiplier():
    numbers = {Decimal("45210300"), Decimal("17.30")}
    assert read_figure("45.210.300", "bin TL", 45_210_300_000, "TL", numbers) == (
        45_210_300_000,
        "TL",
    )
    assert read_figure("%17,30", None, 17.3, "%", numbers) == (17.3, "%")


def test_read_figure_rejects_unit_disagreement_and_absent_numbers():
    numbers = {Decimal("45210300")}
    with pytest.raises(FigureError, match="model value"):
        read_figure("45.210.300", "bin TL", 45_210_300, "TL", numbers)  # forgot "bin"
    with pytest.raises(FigureError, match="not found"):
        read_figure("46.000.000", "bin TL", 46_000_000_000, "TL", numbers)
    with pytest.raises(FigureError, match="not understood"):
        read_figure("45.210.300", "ton CO2", 45_210_300, "adet", numbers)


def test_valid_tablo_draft(page):
    text, numbers = page
    d = validate_draft(out(), TASK, 0, text, numbers)
    assert d.key == "page:akbank-2024:3:0"
    assert d.expected.value == 45_210_300_000 and d.expected.unit == "TL"
    assert d.expected.tolerance.type == "rel" and d.expected.tolerance.amount == 0.01
    assert [(e.doc, e.page) for e in d.evidence] == [("akbank-2024", 3)]
    assert d.banks == ["akbank"] and d.years == [2024] and not d.llm_checked


@pytest.mark.parametrize(
    "kw, reason",
    [
        ({"question": "2024 yılında konsolide net kâr ne kadar?"}, "bank"),
        ({"question": "Akbank konsolide bazda ne kadar net kâr elde etti?"}, "year"),
        ({"question": "Akbank 2024 yılında ne kadar net kâr elde etti?"}, "basis"),
        ({"basis": None}, "basis"),
        ({"year": 2023}, "report year"),
        ({"type": "karsilastirma"}, None),  # rejected by the output model itself
        ({"value": 45_210_300}, "model value"),
        (
            {
                "question": "Akbank 2024 konsolide net kârı, tutarlar aksi belirtilmedikçe bin "
                "Türk Lirası olarak ne kadar?"
            },
            "copies",
        ),
    ],
)
def test_invalid_drafts_are_rejected(page, kw, reason):
    text, numbers = page
    if reason is None:
        with pytest.raises(ValueError):
            out(**kw)
        return
    with pytest.raises(ValueError, match=reason):
        validate_draft(out(**kw), TASK, 0, text, numbers)


def test_hesaplama_change_is_computed_in_python(page):
    text, numbers = page
    d = validate_draft(
        out(
            type="hesaplama",
            question="Akbank'ın konsolide net kârı 2023'ten 2024'e yüzde kaç değişti?",
            answer="2024: 45.210.300; 2023: 38.100.000",
            prior_figure="38.100.000",
            prior_value=38_100_000_000,
        ),
        TASK,
        1,
        text,
        numbers,
    )
    assert d.years == [2023, 2024]
    assert d.expected.value == 18.66  # (45.2103 - 38.1) / 38.1 = 18.66%
    assert d.expected.unit == "%" and d.expected.tolerance.amount == 0.5
    assert "38.100.000" in d.quote


def test_hesaplama_needs_basis_for_money(page):
    text, numbers = page
    draft = out(
        type="hesaplama",
        basis=None,
        prior_figure="38.100.000",
        prior_value=38_100_000_000,
        question="Akbank'ın net kârı 2023'ten 2024'e yüzde kaç değişti?",
    )
    with pytest.raises(ValueError, match="unknown basis"):
        validate_draft(draft, TASK, 1, text, numbers)


def test_sentez_and_text_olgusal(page):
    text, numbers = page
    s = validate_draft(
        out(
            type="sentez",
            basis=None,
            figure=None,
            unit_text=None,
            value=None,
            unit=None,
            question="Akbank 2024'te sürdürülebilirlik alanında hangi adımları attı?",
            answer="...",
            key_points=["yeşil ürünler", "yenilenebilir enerji", ""],
        ),
        TASK,
        2,
        text,
        numbers,
    )
    assert s.key_points == ["yeşil ürünler", "yenilenebilir enerji"] and s.expected.value is None
    o = validate_draft(
        out(
            type="olgusal",
            basis=None,
            figure=None,
            unit_text=None,
            value=None,
            unit=None,
            question="Akbank 2024'te hangi alanda yeni ürünler geliştirdi?",
            answer="...",
        ),
        TASK,
        3,
        text,
        numbers,
    )
    assert o.expected.value is None and o.expected.tolerance is None


def test_validate_page_drops_duplicates_and_unrequested(page):
    text, numbers = page
    task = PageTask("akbank-2024", "akbank", 2024, 3, {"tablo": 2})
    result = PageDrafts(questions=[out(), out(), out(type="sentez", key_points=["a", "b"])])
    drafts, drops = validate_page(result, task, text, numbers)
    assert len(drafts) == 1
    assert [r for _, r, _ in drops] == [
        "duplicate question on the page",
        "type 'sentez' was not requested",
    ]


def test_basis_words_and_copy_check():
    assert bases_mentioned("konsolide olmayan net kâr") == {"solo"}
    assert bases_mentioned("konsolide ve solo") == {"konsolide", "solo"}
    assert bases_mentioned("net kâr") == set()
    assert copied_run("bir iki üç dört beş altı", "x bir iki üç dört beş altı y") is not None
    assert copied_run("bir iki üç dört beş", "bir iki üç dört beş altı") is None


def test_check_marks_passing_drafts_and_drops_the_rest(page):
    text, numbers = page
    a = validate_draft(out(), TASK, 0, text, numbers)
    b = a.model_copy(update={"key": "b"})
    c = a.model_copy(update={"key": "c"})
    check = PageCheck(
        items=[
            CheckItem(
                index=0,
                supported=True,
                figure_ok=True,
                basis_ok=True,
                self_contained=True,
                reason="ok",
            ),
            CheckItem(
                index=1,
                supported=True,
                figure_ok=False,
                basis_ok=True,
                self_contained=True,
                reason="prior-year column",
            ),
        ]
    )
    passed, dropped = apply_check([a, b, c], check)
    assert [d.key for d in passed] == [a.key] and passed[0].llm_checked
    assert [(d.key, r.split(":")[0]) for d, r in dropped] == [
        ("b", "check failed (figure_ok)"),
        ("c", "no check verdict"),
    ]
    none_passed, all_dropped = apply_check([a], None)
    assert none_passed == [] and all_dropped[0][1] == "check call failed"


def test_check_prompt_lists_drafts_and_pages(page):
    text, numbers = page
    d = validate_draft(out(), TASK, 0, text, numbers)
    prompt = build_prompt("akbank", 2024, [(3, text)], [d])
    assert "[0] tip=tablo" in prompt and "45.210.300" in prompt and text in prompt
    multi = build_prompt("akbank", 2024, [(3, text), (4, "x")], [d])
    assert "kanıt sayfası=3" in multi and "Kaynak sayfa 4" in multi


def test_read_figure_compound_amounts_and_components():
    numbers = {Decimal(4), Decimal(548), Decimal(693), Decimal(1)}
    page = "4 trilyon 548 milyar TL aktif; 693 yurt içi ve 1 yurt dışı şube"
    assert read_figure("4 trilyon 548 milyar", "TL", 4_548_000_000_000, "TL", numbers, page) == (
        4_548_000_000_000,
        "TL",
    )
    assert read_figure(None, "şube", 694, "adet", numbers, page, components=["693", "1"]) == (
        694,
        "adet",
    )
    with pytest.raises(FigureError, match="exactly one number"):
        read_figure("548 milyar 4 trilyon", "TL", 1, "TL", numbers, page)


def test_read_figure_refuses_a_guessed_scale():
    numbers = {Decimal("25412766")}
    with pytest.raises(FigureError, match="scale 'bin' is not written"):
        read_figure("25.412.766", "bin TL", 25_412_766_000, "TL", numbers, "Toplam 25.412.766")
    assert read_figure("25.412.766", "bin TL", 25_412_766_000, "TL", numbers, "(Bin TL) x")


def test_olgusal_with_non_canonical_unit_is_kept_as_text(page):
    text, numbers = page
    d = validate_draft(
        out(
            type="olgusal",
            basis=None,
            figure="17,30",
            unit_text="MW",
            value=17.3,
            unit=None,
            question="Akbank 2024'te kaç MW güneş kapasitesi kurdu?",
            answer="17,30 MW",
        ),
        TASK,
        4,
        text,
        numbers,
    )
    assert d.expected.value is None and "scored as text" in d.notes
    with pytest.raises(FigureError, match="not found"):
        validate_draft(
            out(
                type="olgusal",
                basis=None,
                figure="99,9",
                unit_text="MW",
                value=99.9,
                unit=None,
                question="Akbank 2024'te kaç MW güneş kapasitesi kurdu?",
                answer="99,9 MW",
            ),
            TASK,
            4,
            text,
            numbers,
        )


def test_check_prompt_explains_count_metrics_have_no_basis(page):
    text, numbers = page
    d = validate_draft(out(), TASK, 0, text, numbers).model_copy(
        update={"source": "probe", "metric": "branches"}
    )
    assert "Adet metriği" in build_prompt("akbank", 2024, [(3, text)], [d])


# --- fix round 1: unknown basis, figure check required -----------------------------------


def test_unknown_basis_money_and_ratio_singles_are_dropped(page):
    text, numbers = page
    q = "Akbank 2024 yılında ne kadar net kâr elde etti?"
    with pytest.raises(ValueError, match="unknown basis"):
        validate_draft(out(basis=None, question=q), TASK, 0, text, numbers)
    ratio = out(
        basis=None,
        figure="17,30",
        unit_text="%",
        value=17.3,
        unit="%",
        question="Akbank'ın 2024 sermaye yeterliliği oranı yüzde kaçtı?",
    )
    with pytest.raises(ValueError, match="unknown basis"):
        validate_draft(ratio, TASK, 0, text, numbers)
    olgusal = out(type="olgusal", basis=None, question=q)
    with pytest.raises(ValueError, match="unknown basis"):
        validate_draft(olgusal, TASK, 0, text, numbers)


def test_unknown_basis_count_single_is_kept(tmp_dir):
    pages = DocPages.load("akbank-2024", write_parsed(tmp_dir, "akbank-2024"))
    page_text, numbers = pages.render(3), pages.numbers(3) | {Decimal(694)}
    count = out(
        type="olgusal",
        basis=None,
        figure="694",
        unit_text="şube",
        value=694,
        unit="adet",
        question="Akbank 2024 sonunda kaç şubeyle çalışıyordu?",
    )
    d = validate_draft(count, TASK, 0, page_text, numbers)
    assert d.basis is None and d.expected.value == 694 and d.expected.unit == "adet"


def test_validate_page_logs_unknown_basis_drop(page):
    text, numbers = page
    task = PageTask("akbank-2024", "akbank", 2024, 3, {"tablo": 2})
    result = PageDrafts(
        questions=[out(basis=None, question="Akbank 2024 yılında ne kadar net kâr etti?")]
    )
    drafts, drops = validate_page(result, task, text, numbers)
    assert drafts == [] and "unknown basis" in drops[0][1]


@pytest.mark.parametrize("figure_ok", [None, False])
def test_numeric_draft_needs_a_positive_figure_check(page, figure_ok):
    text, numbers = page
    d = validate_draft(out(), TASK, 0, text, numbers)
    check = PageCheck(
        items=[
            CheckItem(
                index=0,
                supported=True,
                figure_ok=figure_ok,
                basis_ok=True,
                self_contained=True,
                reason="r",
            )
        ]
    )
    passed, dropped = apply_check([d], check)
    assert passed == [] and "figure_ok" in dropped[0][1]


def test_text_draft_passes_with_null_figure_check(page):
    text, numbers = page
    d = validate_draft(
        out(
            type="olgusal",
            basis=None,
            figure=None,
            unit_text=None,
            value=None,
            unit=None,
            question="Akbank 2024'te hangi alanda yeni ürünler geliştirdi?",
            answer="...",
        ),
        TASK,
        0,
        text,
        numbers,
    )
    check = PageCheck(
        items=[
            CheckItem(
                index=0,
                supported=True,
                figure_ok=None,
                basis_ok=True,
                self_contained=True,
                reason="r",
            )
        ]
    )
    passed, _ = apply_check([d], check)
    assert len(passed) == 1 and passed[0].llm_checked


def test_tolerance_covers_the_rounding_of_the_printed_figure():
    from kaprag.gold.figures import rounding_tolerance

    def tol(*args):
        t = rounding_tolerance(*args)
        return t.type, t.amount

    assert tol("2,6", "trilyon TL", "TL", 2_600_000_000_000) == ("rel", 0.0193)
    assert tol("111", "milyar TL", "TL", 111_000_000_000) == ("rel", 0.01)  # 0.45% < 1%
    assert tol("1.073.924", "bin TL", "TL", 1_073_924_000) == ("rel", 0.01)
    assert tol("1 trilyon 905 milyar", "TL", "TL", 1_905_000_000_000) == ("rel", 0.01)
    assert tol("17", "%", "%", 17) == ("abs", 0.5)
    assert tol("%17,3", None, "%", 17.3) == ("abs", 0.2)
    assert tol("27,7", "milyon müşteri", "adet", 27_700_000) == ("abs", 50_000)
    assert tol("21.167", "kişi", "adet", 21_167) == ("abs", 0)
    assert tol(None, "şube", "adet", 694) == ("abs", 0)  # summed components are exact
