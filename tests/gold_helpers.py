"""Synthetic parsed pages and a fake LLM client for the gold-drafting tests."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kaprag.parsing.outputs import Element
from kaprag.retrieval import SearchHit

TABLE_MD = (
    "| | Cari Dönem 31 Aralık 2024 | Önceki Dönem 31 Aralık 2023 |\n"
    "|---|---|---|\n"
    "| Net Dönem Kârı | 45.210.300 | 38.100.000 |\n"
    "| Toplam Aktifler | 2.500.000.000 | 1.900.000.000 |\n"
    "| Özkaynaklar | 250.400.000 | 190.300.000 |\n"
    "| Faiz Gelirleri | 400.100.000 | 300.200.000 |\n"
    "| Faiz Giderleri | (210.500.000) | (150.600.000) |\n"
    "| Sermaye Yeterliliği Oranı (%) | 17,30 | 18,20 |\n"
)
PROSE = (
    "Banka, 2024 yılında dijital bankacılık stratejisini sürdürerek mobil kullanıcı sayısını "
    "artırdı. Sürdürülebilir finansman alanında yeni ürünler geliştirildi, yenilenebilir enerji "
    "projelerine kaynak sağlandı ve iklim hedefleri doğrultusunda karbon emisyonları azaltıldı. "
)


def element(page: int, type_: str, text: str, **kw: Any) -> dict:
    row = {
        "page": page,
        "pages": [page],
        "type": type_,
        "label": kw.pop(
            "label",
            {"heading": "section_header", "furniture": "page_footer"}.get(
                type_, type_ if type_ == "table" else "text"
            ),
        ),
        "level": 1 if type_ == "heading" else None,
        "section_path": kw.pop("section_path", ["Finansal Bilgiler"]),
        "text": text,
        "caption": None,
        "in_picture": False,
    }
    row.update(kw)
    return row


def synthetic_doc(year: int = 2024) -> list[dict]:
    """Rows of a 6-page document: cover, TOC, table page, text page, empty page, mixed."""
    table = TABLE_MD.replace("2024", str(year)).replace("2023", str(year - 1))
    return [
        element(1, "heading", "Entegre Faaliyet Raporu"),
        element(2, "heading", "İçindekiler"),
        *[element(2, "list_item", f"Bölüm {i} {10 + i}") for i in range(10)],
        element(3, "heading", f"KONSOLİDE FİNANSAL TABLOLAR {year}"),
        element(
            3,
            "paragraph",
            '(Tutarlar aksi belirtilmedikçe bin Türk Lirası ("TL") olarak ifade edilmiştir.)',
        ),
        *[element(3, "table", table) for _ in range(3)],
        element(3, "furniture", "Sayfa 3"),
        element(4, "heading", "Strateji"),
        element(4, "paragraph", PROSE * 8),
        element(5, "paragraph", "Not"),
        element(6, "paragraph", PROSE + " Tutarlar bin TL."),
        element(6, "table", table),
    ]


def write_parsed(root: Path, doc_id: str, year: int = 2024) -> Path:
    """Write Docling and PyMuPDF files of a synthetic document; returns the parsed dir."""
    rows = synthetic_doc(year)
    for parser in ("docling", "pymupdf"):
        (root / parser).mkdir(parents=True, exist_ok=True)
    (root / "docling" / f"{doc_id}.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    pages: dict[int, list[str]] = {}
    for r in rows:
        pages.setdefault(r["page"], []).append(r["text"])
    plain = [{"page": p, "text": "\n".join(t)} for p, t in sorted(pages.items())]
    (root / "pymupdf" / f"{doc_id}.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in plain), encoding="utf-8"
    )
    return root


def elements(rows: list[dict], page: int) -> list[Element]:
    return [Element(**r) for r in rows if r["page"] == page]


@dataclass
class FakeResult:
    parsed: Any


class FakeClient:
    """Answers `complete` with `responders[stage](user_prompt, schema)`; records calls.

    A responder may raise `kaprag.llm.LLMCallError` to simulate a failed call."""

    def __init__(self, responders: dict[str, Any]) -> None:
        self.responders = responders
        self.calls: list[tuple[str, str]] = []

    async def complete(self, system, user, model, schema=None, *, tags=None):
        stage = (tags or {}).get("stage", "")
        self.calls.append((stage, model))
        payload = self.responders[stage](user, schema)
        return FakeResult(parsed=schema.model_validate(payload))


class FakeSearcher:
    def __init__(self, hits: list[SearchHit]) -> None:
        self.hits = hits
        self.queries: list[tuple[str, Any]] = []

    def search(self, query, doc_ids=None, k=8):
        self.queries.append((query, doc_ids))
        return [h for h in self.hits if doc_ids is None or h.doc_id in doc_ids][:k]
