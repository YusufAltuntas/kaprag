"""End-to-end drafting run on synthetic pages with a fake LLM and a fake searcher."""

import asyncio
import importlib.util
import json
import re
import sys
from pathlib import Path

from gold_helpers import FakeClient, FakeSearcher, write_parsed

from kaprag.corpus import Document
from kaprag.eval.schema import load_questions
from kaprag.gold.guard import overwrite_refusal
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
    """One question per kind, only for the first focus of the kind (Ziraat / Akbank); the
    olmayan_bilgi call also returns a mis-kinded question, which must be dropped."""
    kind = re.search(r"hepsi kind=(\w+)", user).group(1)
    if "Ziraat" not in user and "- Akbank için" not in user:
        return {"questions": []}
    wanted = {kind, "korpus_disi_banka"} if kind == "olmayan_bilgi" else {kind}
    return {"questions": [q for q in ALL_ABSTAIN if q["kind"] in wanted]}


ALL_ABSTAIN = {
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
}["questions"]


def evidence_reply(user, schema):
    """Approves figure-matched pages that show the 2024 table; rejects the rest."""
    items = []
    for i, page in re.findall(r"^### Aday (\d+) \(sayfa (\d+)\)", user, flags=re.M):
        items.append({"candidate": int(i), "states_answer": page == "6", "reason": "r"})
    return {"items": items}


def judge_reply(user, schema):
    items = []
    for i, question in re.findall(r"^### Soru (\d+): (.*)$", user, flags=re.M):
        premise = "zarar" in question
        items.append(
            {
                "index": int(i),
                "verdict": "false_premise" if premise else "not_in_corpus",
                "evidence_passages": [1] if premise else [],
                "reason": "kâr" if premise else "yok",
            }
        )
    return {"items": items}


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


HITS = [SearchHit(f"{d}:c", d, 3, 3, 1.0, "text", (), "t") for d in ("akbank-2024", "isbank-2024")]


def run_pipeline(tmp_dir, searcher):
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
            "evidence": evidence_reply,
        }
    )
    run_dir, out = tmp_dir / "run", tmp_dir / "gold" / "questions.jsonl"
    cfg = RunConfig(
        run_dir=run_dir,
        out=out,
        docs=docs,
        table_pages=1,
        text_pages=1,
        metrics=[METRICS_BY_ID["net_profit"]],
        abstain_per_call=1,
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
    summary = asyncio.run(Pipeline(cfg, client, lambda: searcher).run())
    return summary, run_dir, out


def test_pipeline_end_to_end(tmp_dir):
    summary, run_dir, out = run_pipeline(tmp_dir, FakeSearcher(HITS))

    questions = load_questions(out)
    types = {q.type for q in questions}
    assert types == {"tablo", "hesaplama", "karsilastirma", "sentez", "cevapsiz", "belirsiz"}
    assert all(q.author == "llm" and not q.verified for q in questions)
    assert all(q.llm_checked != (q.expected_behavior == "clarify") for q in questions)
    ids = {q.id for q in questions}
    assert all(set(q.derived_from) <= ids for q in questions)
    gap = next(q for q in questions if q.type == "hesaplama" and len(q.banks) == 2)
    assert gap.expected.value == 18.66  # Akbank 45.2103 bn vs İş Bankası 38.1 bn
    dev = sum(q.split == "dev" for q in questions)
    assert sum(q.pilot for q in questions) == min(3, dev)
    assert all(q.split == "dev" for q in questions if q.pilot)

    counts = summary["counts"]
    assert counts["failures_by_stage"] == {"draft": 1}  # one page failed; run went on
    assert counts["drops_by_stage"]["draft_validation"] == 2
    assert counts["drops_by_stage"]["abstain_validation"] == 2  # mis-kinded, banks mismatch
    assert counts["drops_by_stage"]["dedup"] == 1  # Akbank page net profit == its probe
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

    # evidence completion: page 6 repeats the table -> added after the original page
    tablo = [q for q in questions if q.type == "tablo"]
    assert tablo and all([e.page for e in q.evidence] == [3, 6] for q in tablo)
    gap = next(q for q in questions if q.type == "hesaplama" and len(q.banks) == 2)
    assert {(e.doc, e.page) for e in gap.evidence} == {
        (b, p) for b in ("akbank-2024", "isbank-2024") for p in (3, 6)
    }
    records = [
        json.loads(x) for x in (run_dir / "evidence.jsonl").read_text("utf-8").split("\n") if x
    ]
    assert records and any(r["added"] for r in records)
    for r in records:  # only pages that passed the check are added; the rest are logged
        assert [a["page"] for a in r["added"]] == [
            c["page"] for c in r["candidates"] if c["page"] == 6
        ]
        assert len(r["added"]) + len(r["rejected"]) == len(r["candidates"])


class FailingSearcher(FakeSearcher):
    """Fails for whole-corpus queries and for isbank-2024 probe queries."""

    def search(self, query, doc_ids=None, k=8):
        if doc_ids is None or doc_ids == ["isbank-2024"]:
            raise RuntimeError("database went away")
        return super().search(query, doc_ids, k)


def test_search_failure_is_logged_and_the_run_continues(tmp_dir):
    summary, run_dir, out = run_pipeline(tmp_dir, FailingSearcher(HITS))
    failures = [
        json.loads(x) for x in (run_dir / "failures.jsonl").read_text("utf-8").split("\n") if x
    ]
    search_failures = [f for f in failures if f["stage"].endswith("search")]
    assert {f["stage"] for f in search_failures} == {
        "probe_search",
        "abstain_search",
        "evidence_search",
    }
    assert all("database went away" in f["error"] for f in search_failures)
    questions = load_questions(out)
    assert not any("Ziraat" in q.question for q in questions)  # its corpus check never ran
    assert any("zarar" in q.question for q in questions)  # bank-filtered search still worked
    assert not any(
        q.banks == ["isbank"]
        and q.type == "tablo"
        and q.derived_from == []
        and "net kâr elde" in q.question
        and q.notes.startswith("probe")
        for q in questions
    )


# --- overwrite guard (scripts/draft_gold.py --out / --overwrite) ---------------------------

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "draft_gold.py"


def _gold_row(qid, *, verified=False, author="llm"):
    return json.dumps({"id": qid, "author": author, "verified": verified}) + "\n"


def _draft_gold_main(monkeypatch, argv):
    spec = importlib.util.spec_from_file_location("draft_gold_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def no_llm(*_, **__):
        raise AssertionError("the guard must run before any LLM client is created")

    monkeypatch.setattr(module, "create_client", no_llm)
    monkeypatch.setattr(module, "table_complete", no_llm)
    monkeypatch.setattr(sys, "argv", ["draft_gold.py", *argv])
    return module.main()


def test_overwrite_guard_allows_a_fresh_path(tmp_dir):
    assert overwrite_refusal(tmp_dir / "new.jsonl", overwrite=False) is None
    assert overwrite_refusal(tmp_dir / "new.jsonl", overwrite=True) is None


def test_overwrite_guard_refuses_an_existing_file_without_the_flag(tmp_dir):
    out = tmp_dir / "questions.jsonl"
    out.write_text(_gold_row("q1"), encoding="utf-8")
    assert "--overwrite" in overwrite_refusal(out, overwrite=False)
    assert overwrite_refusal(out, overwrite=True) is None  # drafts only: the flag allows it


def test_overwrite_guard_refuses_verified_or_human_rows_even_with_the_flag(tmp_dir):
    verified = tmp_dir / "verified.jsonl"
    verified.write_text(_gold_row("q1") + _gold_row("q2", verified=True), encoding="utf-8")
    human = tmp_dir / "human.jsonl"
    human.write_text(_gold_row("q1") + "\n" + _gold_row("q3", author="human"), encoding="utf-8")
    for path in (verified, human):
        for flag in (False, True):
            assert overwrite_refusal(path, overwrite=flag) is not None
    assert "q2" in overwrite_refusal(verified, overwrite=True)
    assert "q3" in overwrite_refusal(human, overwrite=True)


def test_draft_gold_cli_exits_2_before_any_llm_call(tmp_dir, monkeypatch, capsys):
    out = tmp_dir / "questions.jsonl"
    out.write_text(_gold_row("q1", verified=True), encoding="utf-8")
    before = out.read_bytes()
    argv = [str(tmp_dir / "run"), "--out", str(out)]
    assert _draft_gold_main(monkeypatch, argv) == 2
    assert _draft_gold_main(monkeypatch, [*argv, "--overwrite"]) == 2
    assert out.read_bytes() == before
    assert not (tmp_dir / "run").exists()
    assert "verified" in capsys.readouterr().err
