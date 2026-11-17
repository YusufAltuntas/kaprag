"""The indexing-time role rule: headings, statutory sections, page map, chunk roles."""

import pytest

from kaprag.access.roles import FOOTNOTES, PUBLIC, RISK
from kaprag.access.rules import (
    DocumentAccess,
    chunk_roles,
    fold,
    heading_roles,
    path_roles,
    statutory_section,
)
from kaprag.chunking.base import Chunk
from kaprag.corpus import Document
from kaprag.index.store import access_roles_for
from kaprag.parsing.outputs import Element


def el(page, text, type_="paragraph", path=(), pages=None, in_picture=False):
    return Element(
        page=page,
        pages=list(pages or [page]),
        type=type_,
        label=type_,
        level=1 if type_ == "heading" else None,
        section_path=list(path),
        text=text,
        caption=None,
        in_picture=in_picture,
    )


def h(page, text, path=()):
    return el(page, text, "heading", path)


def test_fold_is_turkish_case_insensitive():
    assert fold("RİSK YÖNETİMİ") == fold("RISK YÖNETIMI") == fold("Risk yönetimi")
    assert fold("Risk  Yönetimi\n") == "risk yönetimi"
    assert fold("DİPNOTLAR") == "dipnotlar"


@pytest.mark.parametrize(
    "heading",
    [
        "Risk Yönetimi",
        "RİSK YÖNETİMİ",
        "RISK YÖNETIMI",  # Docling often loses the dot of İ
        "Risk yönetimi ve risk ağırlıklı tutarlara ilişkin genel açıklamalar",
        "XI. Risk Yönetimine İlişkin Açıklamalar",
        "Finansal Bilgiler ve Risk Yönetimi",
    ],
)
def test_risk_management_headings(heading):
    assert heading_roles(heading) == RISK


@pytest.mark.parametrize(
    "heading",
    [
        "Kredi Riski",
        "Risk Yönetim Politikaları",
        "Sürdürülebilirlik",
        "Trading Bölümü",
        "",
        # footnotes come from statutory section five only, not from a heading phrase:
        # this running title repeats over all seven sections in some reports
        "31 Aralık 2023 tarihi itibarıyla finansal tablolara ilişkin açıklama ve dipnotlar",
        "IV. Gelir Tablosuna İlişkin Açıklama ve Dipnotlar",
    ],
)
def test_other_headings_are_public(heading):
    assert heading_roles(heading) == PUBLIC


def test_any_risk_heading_along_the_path_counts():
    assert path_roles(["Finansal tablolara ilişkin dipnotlar", "Kredi Riski"]) == PUBLIC
    assert path_roles(["Risk Yönetimi", "Kredi Riski"]) == RISK  # under a risk heading
    assert path_roles(["Hakkımızda", "RİSK YÖNETİMİ", "Kredi Riski"]) == RISK
    assert path_roles(["Hakkımızda", "Strateji"]) == PUBLIC
    assert path_roles([]) == PUBLIC


@pytest.mark.parametrize(
    ("heading", "number"),
    [
        ("BİRİNCİ BÖLÜM BANKA HAKKINDA GENEL BİLGİLER", 1),
        ("İKİNCİ BÖLÜM", 2),
        ("ÜÇÜNCÜ BÖLÜM: MUHASEBE POLİTİKALARINA İLİŞKİN AÇIKLAMALAR", 3),
        ("DÖRDÜNCÜ BÖLÜM", 4),
        ("Dördüncü Bölüm - Mali bünyeye ve risk yönetimine ilişkin bilgiler", 4),
        ("BEŞİNCİ BÖLÜMs", 5),
        ("ALTINCI BÖLÜM DİĞER AÇIKLAMALAR", 6),
        ("YEDİNCİ BÖLÜM BAĞIMSIZ DENETÇİ RAPORUNA İLİŞKİN AÇIKLAMALAR", 7),
        ("Yapı ve Kredi Bankası A.Ş. Üçüncü Bölüm", 3),
        ("Beşinci Bölüm - Konsolide finansal tablolara ilişkin açıklama ve dipnotlar", 5),
        ("Trading Bölümü", None),
        ("İlgili Rapor Bölümü", None),
        ("Kredi Riski", None),
        # cross-references in running text are not titles
        ("Bu konuda Dördüncü Bölüm'de yer alan X no.lu dipnotta açıklama yapılmıştır.", None),
        ("Dördüncü Bölüm X no.lu dipnotta açıklama yapılmıştır.", None),
        ("Dördüncü Bölüm II no.lu dipnotlara bakınız.", None),
        ("Üçüncü Bölüm VII-b no'lu dipnotta açıklandığı üzere", None),
        ("Beşinci Bölüm II.ğ.4.4 no.lu dipnotta", None),
        ("Beşinci Bölüm II-h numaralı dipnotta açıklanan devir", None),
        ("Üçüncü Bölüm 19 no'lu dipnotta detaylı olarak açıklanmıştır.", None),
        ("Dördüncü Bölüm'de açıklanmıştır.", None),
        ("Dördüncü Bölüm, 7 no'lu dipnotta açıklanmıştır.", None),
        ("7.1. Gerçeğe uygun değer farkı: Dördüncü Bölüm - Konsolide bazda mali bünye", None),
        ("| Dipnot (Beşinci Bölüm) | TP |", None),
        # a Turkish suffix on "bölüm" marks running text, not a title
        ("Birinci bölümde açıklanan esaslar çerçevesinde", None),
        ("Beşinci bölümün ilgili dipnotlarında", None),
        ("Üçüncü bölümden itibaren", None),
        ("Dördüncü bölüme ilişkin açıklamalar", None),
        ("İkinci bölümle birlikte", None),
        ("Beşinci bölümler halinde", None),
        ("BEŞİNCİ BÖLÜMÜ", None),
    ],
)
def test_statutory_section_titles(heading, number):
    assert statutory_section(heading) == number


def statutory_doc():
    """Pages 1-2 narrative (a risk chapter on page 2), 3 statements, 4-5 section four,
    6-7 section five, 8 section six."""
    return [
        h(1, "Hakkımızda"),
        el(1, "Banka 1924'te kuruldu.", path=["Hakkımızda"]),
        el(1, "Sayfa 1", "furniture"),
        h(2, "Risk Yönetimi"),
        el(2, "Kredi riski komitesi.", path=["Risk Yönetimi"]),
        h(2, "Kredi Riski", path=["Risk Yönetimi"]),
        el(2, "Limitler.", path=["Risk Yönetimi", "Kredi Riski"]),
        h(3, "İKİNCİ BÖLÜM FİNANSAL TABLOLAR"),
        el(3, "| Aktif | 100 |", "table", path=["İKİNCİ BÖLÜM FİNANSAL TABLOLAR"]),
        h(4, "DÖRDÜNCÜ BÖLÜM MALİ BÜNYEYE İLİŞKİN BİLGİLER"),
        h(4, "II. Kredi Riskine İlişkin Açıklamalar", path=["DÖRDÜNCÜ BÖLÜM"]),
        el(4, "Sektörel dağılım.", path=["II. Kredi Riskine İlişkin Açıklamalar"]),
        el(5, "Likidite karşılama oranı.", path=["VI. Likidite"]),
        el(6, "Kur riski tablosu sonu.", path=["VI. Likidite"]),
        h(6, "BEŞİNCİ BÖLÜM"),
        h(6, "I. Bilançonun aktif hesaplarına ilişkin açıklamalar"),
        el(6, "Nakit değerler.", path=["I. Bilançonun aktif hesaplarına ilişkin açıklamalar"]),
        el(7, "Kredilerin vade yapısı.", path=["b. Krediler"], pages=[7]),
        h(8, "ALTINCI BÖLÜM DİĞER AÇIKLAMALAR"),
        el(8, "Derecelendirme notları.", path=["ALTINCI BÖLÜM DİĞER AÇIKLAMALAR"]),
        el(8, "Bölüm beş içeriği resimde", path=["b. Krediler"], in_picture=True),
    ]


def test_page_roles_combine_headings_and_statutory_sections():
    access = DocumentAccess.from_elements(statutory_doc())
    assert access.page_roles == {
        1: PUBLIC,
        2: RISK,  # risk chapter by heading
        3: PUBLIC,  # section two: the statements themselves
        4: RISK,  # section four ("mali bünye ve risk yönetimi"), whatever the subheading
        5: RISK,
        6: RISK,  # section four ends and section five starts on this page
        7: FOOTNOTES,
        8: PUBLIC,  # section six; the picture row's heading path is not restricted
    }
    assert access.section_roles == {
        1: PUBLIC, 2: PUBLIC, 3: PUBLIC, 4: RISK, 5: RISK, 6: RISK, 7: FOOTNOTES, 8: PUBLIC,
    }  # fmt: skip


def test_a_section_title_in_a_paragraph_row_opens_the_section():
    # Docling sometimes labels a section title as a paragraph; furniture never opens one
    access = DocumentAccess.from_elements(
        [
            el(1, "DÖRDÜNCÜ BÖLÜM: MALİ BÜNYEYE VE RİSK YÖNETİMİNE İLİŞKİN BİLGİLER"),
            el(1, "Sermaye yeterliliği.", path=["I. Özkaynak"]),
            el(2, "BEŞİNCİ BÖLÜM: FİNANSAL TABLOLARA İLİŞKİN AÇIKLAMA VE DİPNOTLAR"),
            el(3, "Krediler.", path=["a. Krediler"]),
            el(3, "Bu konuda Dördüncü Bölüm'de yer alan X no.lu dipnotta açıklama vardır."),
            el(4, "Ödünç alınan fonlar.", path=["b. Fonlar"]),
            el(4, "ALTINCI BÖLÜM sayfa başlığı", "furniture"),
            el(5, "Diğer.", path=["c. Diğer"]),
        ]
    )
    assert access.page_roles == {1: RISK, 2: FOOTNOTES, 3: FOOTNOTES, 4: FOOTNOTES, 5: FOOTNOTES}


def test_a_row_spanning_pages_marks_all_of_them():
    access = DocumentAccess.from_elements(
        [h(1, "Risk Yönetimi"), el(1, "uzun tablo", "table", ["Risk Yönetimi"], pages=[1, 2, 3])]
    )
    assert access.page_roles == {1: RISK, 2: RISK, 3: RISK}


def test_empty_document_has_no_restricted_pages():
    access = DocumentAccess.from_elements([])
    assert access.page_roles == {}
    assert access.pages_roles(1, 5) == PUBLIC


def test_fixed_chunk_takes_the_most_restrictive_page():
    access = DocumentAccess.from_elements(statutory_doc())
    assert chunk_roles("fixed", (), "text", 1, 1, access) == PUBLIC
    assert chunk_roles("fixed", (), "text", 1, 2, access) == RISK  # public + risk page
    assert chunk_roles("fixed", (), "text", 7, 8, access) == FOOTNOTES  # footnote + public
    assert chunk_roles("fixed", (), "text", 6, 7, access) == RISK  # risk + footnote: risk
    assert chunk_roles("fixed", (), "text", 9, 9, access) == PUBLIC  # page without rows


def test_structure_chunk_uses_its_headings_and_its_statutory_section():
    access = DocumentAccess.from_elements(statutory_doc())
    assert chunk_roles("structure", ("Hakkımızda",), "text", 1, 1, access) == PUBLIC
    # heading path only: a public chunk on a page that also carries risk content stays public
    assert chunk_roles("structure", ("Hakkımızda",), "text", 2, 2, access) == PUBLIC
    assert chunk_roles("structure", ("Risk Yönetimi", "Kredi Riski"), "text", 2, 2, access) == RISK
    assert chunk_roles("structure", ("VI. Likidite",), "text", 5, 5, access) == RISK  # section 4
    assert chunk_roles("structure", ("b. Krediler",), "table", 7, 7, access) == FOOTNOTES
    assert chunk_roles("structure", ("ALTINCI BÖLÜM",), "text", 8, 8, access) == PUBLIC


def test_structure_picture_text_uses_the_page_map():
    # a picture-text chunk carries the path of its page's first picture row only
    access = DocumentAccess.from_elements(
        [
            h(1, "Hakkımızda"),
            el(1, "KPI", path=["Hakkımızda"], in_picture=True),
            h(1, "Risk Yönetimi"),
            el(1, "Risk iştahı 3", path=["Risk Yönetimi"], in_picture=True),
        ]
    )
    assert chunk_roles("structure", ("Hakkımızda",), "picture_text", 1, 1, access) == RISK
    assert chunk_roles("structure", ("Hakkımızda",), "text", 1, 1, access) == PUBLIC


def test_unknown_variant_is_refused():
    with pytest.raises(ValueError, match="variant"):
        chunk_roles("semantic", (), "text", 1, 1, DocumentAccess.from_elements([]))


def chunk(page_start, page_end, path=(), chunk_type="text"):
    return Chunk("akbank-2023", 0, page_start, page_end, chunk_type, tuple(path), "t", 1)


DOC = Document("akbank-2023", "akbank", 2023, "https://example.test", None, None, "")


def test_access_roles_for_is_wired_for_both_variants():
    access = DocumentAccess.from_elements(statutory_doc())
    assert access_roles_for(chunk(1, 1), DOC, "fixed", access) == ["public"]
    assert access_roles_for(chunk(1, 2), DOC, "fixed", access) == ["risk"]
    assert access_roles_for(chunk(7, 7), DOC, "fixed", access) == ["analyst", "risk"]
    assert access_roles_for(chunk(2, 2, ["Hakkımızda"]), DOC, "structure", access) == ["public"]
    assert access_roles_for(chunk(2, 2, ["Risk Yönetimi"]), DOC, "structure", access) == ["risk"]
    assert access_roles_for(chunk(7, 7, ["b. Krediler"]), DOC, "structure", access) == [
        "analyst",
        "risk",
    ]
