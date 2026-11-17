"""End-to-end drafting run on synthetic pages with a fake LLM and a fake searcher."""

import asyncio
import json
import re

from gold_helpers import FakeClient, FakeSearcher, write_parsed

from kaprag.corpus import Document
from kaprag.eval.schema import load_questions
from kaprag.gold.metrics import METRICS_BY_ID
from kaprag.gold.pipeline import Pipeline, RunConfig
from kaprag.llm import LLMCallError, Usage
from kaprag.retrieval import SearchHit

NAMES = {"Akbank": "akbank", "İş Bankası": "isbank"}


def bank_of(prompt: str) -> str:
    return next(name for name in NAMES if f"Banka: {name}" in prompt)


def draft_reply(user, schema):
    name = bank_of(user)
    if "[Tablo]" not in user:  # text page
        if name == "İş Bankası":
            raise LLMCallError("overloaded", attempts=3, usage=Usage())
        return {
            "questions": [
                {
                    "type": "sentez",
                    "question": f"{name} 2024'te sürdürülebilirlikte neler yaptı?",
                    "answer": "...",
                    "year": 2024,
                    "basis": None,
                    "figure": None,
                    "unit_text": None,
                    "value": None,
                    "unit": None,
                    "prior_figure": None,
                    "prior_value": None,
                    "key_points": ["yeşil ürünler", "yenilenebilir enerji"],
                }
            ]
        }
    base = {
        "answer": "45.210.300 bin TL",
        "year": 2024,
        "basis": "konsolide",
        "figure": "45.210.300",
        "unit_text": "bin TL",
        "value": 45_210_300_000,
        "unit": "TL",
        "prior_figure": None,
        "prior_value": None,
        "key_points": [],
    }
    return {
        "questions": [
            base
            | {
                "type": "tablo",
                "question": f"{name} 2024 yılında konsolide bazda ne kadar net kâr elde etti?",
            },
            base
            | {
                "type": "hesaplama",
                "prior_figure": "38.100.000",
                "prior_value": 38_100_000_000,
                "question": f"{name} konsolide net kârı 2023'ten 2024'e yüzde kaç değişti?",
            },
            base
            | {
                "type": "tablo",
                "figure": "123.456",
                "value": 123_456_000,  # not on the page
                "question": f"{name} 2024 yılında konsolide faiz geliri ne kadardı?",
            },
        ]
    }


def check_reply(user, schema):
    n = len(re.findall(r"^\[\d+\] tip=", user, flags=re.M))
    return {
        "items": [
            {
                "index": i,
                "supported": True,
                "figure_ok": True,
                "basis_ok": True,
                "self_contained": True,
                "reason": "ok",
            }
            for i in range(n)
        ]
    }


def probe_reply(user, schema):
    figure = "45.210.300" if bank_of(user) == "Akbank" else "38.100.000"
    return {
        "items": [
            {
                "metric": "net_profit",
                "found": True,
                "passage": 1,
                "figure": figure,
                "components": [],
                "unit_text": "bin TL",
                "value": int(figure.replace(".", "")) * 1000,
                "unit": "TL",
                "basis": "konsolide",
                "answer": f"{figure} bin TL",
            }
        ]
    }


def abstain_reply(user, schema):
    return {
        "questions": [
            {
                "kind": "korpus_disi_banka",
                "question": "Ziraat Bankası 2024'te kaç şubesi vardı?",
                "banks": [],
                "years": [2024],
                "why": "Ziraat korpusta yok",
            },
            {
                "kind": "yanlis_oncul",
                "question": "Akbank 2024 yılında ne kadar zarar etti?",
                "banks": ["akbank"],
                "years": [2024],
                "why": "Akbank kâr etti",
            },
            {
                "kind": "olmayan_bilgi",
                "question": "Akbank'ın Kadıköy şubesinin müşteri sayısı?",
                "banks": ["isbank"],
                "years": [],
                "why": "-",
            },  # banks mismatch -> dropped
        ]
    }


def judge_reply(user, schema):
    return {
        "items": [
            {"index": 0, "verdict": "not_in_corpus", "evidence_passages": [], "reason": "yok"},
            {"index": 1, "verdict": "false_premise", "evidence_passages": [1], "reason": "kâr"},
        ]
    }


def ambiguous_reply(user, schema):
    items = []
    for i, how, question in re.findall(r"\[(\d+)\] Talimat: (.*?)\nSoru: (.*)", user):
        name = next(n for n in NAMES if n in question)
        if how.startswith("Banka"):
            new = question.replace(f"{name}, ", "").replace(name, "")
        elif how.startswith("Yılı"):
            new = question.replace("2024 yılında ", "")
        else:
            new = question.replace("konsolide bazda ", "")
        items.append({"index": int(i), "question": new})
    return {"items": items}


def test_pipeline_end_to_end(tmp_dir):
    parsed = tmp_dir / "parsed"
    for doc in ("akbank-2024", "isbank-2024"):
        write_parsed(parsed, doc)
    docs = [Document(f"{b}-2024", b, 2024, "u", None, None, "") for b in ("akbank", "isbank")]
    client = FakeClient(
        {
            "draft": draft_reply,
            "check": check_reply,
            "probe_extract": probe_reply,
            "probe_check": check_reply,
            "abstain_draft": abstain_reply,
            "abstain_judge": judge_reply,
            "ambiguous": ambiguous_reply,
        }
    )
    hits = [
        SearchHit(f"{d}:c", d, 3, 3, 1.0, "text", (), "t") for d in ("akbank-2024", "isbank-2024")
    ]
    run_dir, out = tmp_dir / "run", tmp_dir / "gold" / "questions.jsonl"
    cfg = RunConfig(
        run_dir=run_dir,
        out=out,
        docs=docs,
        table_pages=1,
        text_pages=1,
        metrics=[METRICS_BY_ID["net_profit"]],
        abstain_per_kind=1,
        ambiguous_per_kind=1,
        quotas={
            "olgusal": 5,
            "tablo": 5,
            "hesaplama": 5,
            "karsilastirma": 5,
            "sentez": 5,
            "cevapsiz": 5,
            "belirsiz": 5,
        },
        pilot=3,
        parsed_dir=parsed,
    )
    summary = asyncio.run(Pipeline(cfg, client, lambda: FakeSearcher(hits)).run())

    questions = load_questions(out)
    types = {q.type for q in questions}
    assert types == {"tablo", "hesaplama", "karsilastirma", "sentez", "cevapsiz", "belirsiz"}
    assert all(q.author == "llm" and not q.verified for q in questions)
    assert all(q.llm_checked != (q.expected_behavior == "clarify") for q in questions)
    ids = {q.id for q in questions}
    assert all(set(q.derived_from) <= ids for q in questions)
    gap = next(q for q in questions if q.type == "hesaplama" and len(q.banks) == 2)
    assert gap.expected.value == 18.66  # Akbank 45.2103 bn vs İş Bankası 38.1 bn
    assert sum(q.pilot for q in questions) == 3

    counts = summary["counts"]
    assert counts["failures_by_stage"] == {"draft": 1}  # one page failed; run went on
    assert counts["drops_by_stage"]["draft_validation"] == 2
    assert counts["drops_by_stage"]["abstain_validation"] == 1
    for name in (
        "pages.jsonl",
        "page_drafts.jsonl",
        "page_checked.jsonl",
        "probes.jsonl",
        "derived.jsonl",
        "abstain.jsonl",
        "ambiguous.jsonl",
        "drops.jsonl",
        "failures.jsonl",
        "summary.json",
        "probe_passages.jsonl",
    ):
        assert (run_dir / name).is_file(), name
    drops = [
        json.loads(line)
        for line in (run_dir / "drops.jsonl").read_text("utf-8").split("\n")
        if line
    ]
    assert any("not found on the page" in d["reason"] for d in drops)
