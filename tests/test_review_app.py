"""The Streamlit review app (scripts/review_app.py), against fixture copies only.

Skipped without the `app` dependency group.
"""

import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from kaprag.eval.review import rejected_path_for  # noqa: E402
from kaprag.eval.schema import Question, load_questions, save_questions  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "review_app.py"
FIXTURE = Path(__file__).parent / "fixtures" / "eval" / "questions.jsonl"
ENV_VAR = "KAPRAG_REVIEW_QUESTIONS"


@pytest.fixture
def gold(tmp_dir, monkeypatch):
    path = tmp_dir / "questions.jsonl"
    shutil.copyfile(FIXTURE, path)
    monkeypatch.setenv(ENV_VAR, str(path))
    return path


def run_app() -> AppTest:
    at = AppTest.from_file(str(SCRIPT), default_timeout=30)
    at.run()
    assert not at.exception, at.exception
    return at


def test_app_renders_first_unverified_question(gold):
    at = run_app()
    assert at.selectbox(key="current_qid").value == "f003"
    assert at.selectbox(key="current_qid").options[0].startswith("f003")
    assert any("f003" in h.value for h in at.subheader)
    assert at.metric[0].value == "4 / 13"
    assert gold.read_bytes() == FIXTURE.read_bytes()  # rendering writes nothing


def test_app_approve_button_verifies_one_question(gold):
    at = run_app()
    at.button(key="approve_f003").click().run()
    assert not at.exception
    after = {q.id: q for q in load_questions(gold)}
    before = {q.id: q for q in load_questions(FIXTURE)}
    changed = [qid for qid in before if after[qid] != before[qid]]
    assert changed == ["f003"]
    assert after["f003"].verified is True
    assert at.selectbox(key="current_qid").value == "f004"  # moved on


def test_app_edit_invalid_shows_error_and_writes_nothing(gold):
    at = run_app()
    at.button(key="edit_f003").click().run()
    at.text_input(key="edit_years").set_value("yirmi").run()
    at.button(key="save").click().run()
    assert not at.exception
    assert any("Kaydedilmedi" in e.value for e in at.error)
    assert gold.read_bytes() == FIXTURE.read_bytes()


def test_app_edit_valid_saves_without_verifying(gold):
    at = run_app()
    at.selectbox(key="current_qid").set_value("f008").run()
    at.button(key="edit_f008").click().run()
    at.text_area(key="edit_notes").set_value("Soru metni netleştirildi").run()
    at.button(key="save").click().run()
    assert not at.exception
    after = {q.id: q for q in load_questions(gold)}
    before = {q.id: q for q in load_questions(FIXTURE)}
    assert [qid for qid in before if after[qid] != before[qid]] == ["f008"]
    expected = before["f008"].model_copy(update={"notes": "Soru metni netleştirildi"})
    assert after["f008"] == expected  # basis/unit/tolerance stay None, still unverified


def test_app_reject_moves_question(gold):
    at = run_app()
    at.text_input(key="reject_reason_f003").set_value("yanlış sayfa").run()
    at.button(key="reject_f003").click().run()
    assert not at.exception
    assert "f003" not in {q.id for q in load_questions(gold)}
    assert [q.id for q in load_questions(rejected_path_for(gold))] == ["f003"]


def test_app_shows_audit_status_read_only_and_approve_keeps_it(gold):
    audit = {"status": "fixed", "by": "agent", "at": "2026-11-18", "note": "value corrected"}
    questions = load_questions(gold)
    questions[2] = Question.model_validate(questions[2].model_dump(mode="json") | {"audit": audit})
    assert questions[2].id == "f003"
    save_questions(questions, gold)
    at = run_app()
    text = " ".join(m.value for m in at.markdown)
    assert "Agent denetimi:** düzeltildi (2026-11-18) — value corrected" in text
    assert any("insan doğrulaması değildir" in c.value for c in at.caption)
    at.selectbox(key="current_qid").set_value("f004").run()
    assert "Agent denetimi:** denetlenmedi" in " ".join(m.value for m in at.markdown)
    at.selectbox(key="current_qid").set_value("f003").run()
    at.button(key="approve_f003").click().run()
    assert not at.exception
    after = {q.id: q for q in load_questions(gold)}["f003"]
    assert after.verified is True and after.audit.model_dump() == audit


def test_app_empty_state_and_add_question(tmp_dir, monkeypatch):
    path = tmp_dir / "gold" / "questions.jsonl"
    monkeypatch.setenv(ENV_VAR, str(path))
    at = run_app()
    assert any("Henüz soru yok" in i.value for i in at.info)
    at.radio(key="mode").set_value("Yeni soru ekle").run()
    at.text_area(key="add_question").set_value("Akbank'ın 2024 sonunda kaç şubesi vardı?")
    at.multiselect(key="add_banks").set_value(["akbank"])
    at.text_input(key="add_years").set_value("2024")
    at.text_area(key="add_answer").set_value("700 şube")
    at.text_area(key="add_evidence").set_value("akbank-2024 30")
    at.run()
    at.button(key="add_submit").click().run()
    assert not at.exception
    [added] = load_questions(path)
    assert (added.id, added.author, added.verified) == ("q001", "human", True)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_app_starts_headless_and_answers_http_200(gold, tmp_dir):
    port = free_port()
    log = tmp_dir / "streamlit.log"
    cmd = [
        sys.executable, "-m", "streamlit", "run", str(SCRIPT),
        "--server.headless", "true",
        "--server.address", "127.0.0.1",
        "--server.port", str(port),
        "--browser.gatherUsageStats", "false",
        "--", "--questions", str(gold),
    ]  # fmt: skip
    with log.open("wb") as out:
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT)
        try:
            base = f"http://127.0.0.1:{port}"
            deadline = time.monotonic() + 60
            health = None
            while time.monotonic() < deadline and proc.poll() is None:
                try:
                    health = httpx.get(f"{base}/_stcore/health", timeout=2)
                    break
                except httpx.TransportError:
                    time.sleep(0.5)
            assert health is not None, log.read_text(encoding="utf-8", errors="replace")
            assert health.status_code == 200
            assert httpx.get(base, timeout=10).status_code == 200
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=20)
    assert proc.poll() is not None
