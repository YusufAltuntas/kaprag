"""Gold set review app: approve, edit or reject drafted questions and add your own.

Run (needs the `app` dependency group: `uv sync --group ml --group app`):
  uv run streamlit run scripts/review_app.py
  uv run streamlit run scripts/review_app.py -- --questions path/to/questions.jsonl

The questions file defaults to eval/gold/questions.jsonl; `--questions` (after `--`)
or the KAPRAG_REVIEW_QUESTIONS environment variable points it elsewhere. Rejected
questions go to rejected.jsonl next to it. All writes go through kaprag.eval.review;
this script is only the UI.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import streamlit as st

from kaprag.corpus import DEFAULT_MANIFEST, raw_path, read_manifest, render_page_png
from kaprag.eval.review import (
    ReviewError,
    StaleQuestionError,
    add_human_question,
    approve,
    edit,
    format_evidence,
    load_for_review,
    parse_evidence,
    parse_number,
    parse_years,
    progress,
    reject,
    rejected_path_for,
)
from kaprag.eval.schema import (
    BEHAVIORS,
    DEFAULT_QUESTIONS,
    QUESTION_TYPES,
    GoldSetError,
    Question,
)

ENV_VAR = "KAPRAG_REVIEW_QUESTIONS"
BANKS = ("garanti", "akbank", "isbank", "yapikredi")
SPLITS = ("dev", "test")
AUTHORS = ("llm", "human")
# A selectbox shows a None value as "nothing selected", so empty choices use NONE.
NONE = "—"
BASES = (NONE, "konsolide", "solo")
UNITS = (NONE, "TL", "%", "adet")
TOLERANCE_TYPES = (NONE, "rel", "abs")
BEHAVIOR_LABELS = {
    "answer": "answer: cevapla",
    "abstain": "abstain: bilgi korpusta yok, reddet",
    "clarify": "clarify: netleştirme iste",
    "answer_with_assumption": "answer_with_assumption: varsayımı söyleyerek cevapla",
}
ADD_HELP = (
    "Buradan eklediğiniz soru `author: human` olarak kaydedilir ve **doğrulanmış** sayılır "
    "(`verified: true`): soruyu PDF sayfasına bakarak siz yazdığınız için doğrulama budur. "
    "Kimlik (`qNNN`) ve dev/test ataması otomatik yapılır."
)
_MD_SPECIAL = re.compile(r"([\\`*_\[\]#<>|~$])")
ss = st.session_state


# --- setup -------------------------------------------------------------------------


def questions_path() -> Path:
    """`--questions` after `--`, else $KAPRAG_REVIEW_QUESTIONS, else the real gold file."""
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--questions", type=Path)
    args, _ = parser.parse_known_args(sys.argv[1:])
    if args.questions:
        return args.questions.resolve()
    if os.environ.get(ENV_VAR):
        return Path(os.environ[ENV_VAR]).resolve()
    return DEFAULT_QUESTIONS


@st.cache_data(show_spinner=False)
def doc_ids() -> list[str]:
    try:
        return [d.doc_id for d in read_manifest(DEFAULT_MANIFEST)]
    except (OSError, ValueError):
        return [f"{bank}-{year}" for bank in BANKS for year in (2023, 2024, 2025)]


@st.cache_data(show_spinner=False, max_entries=8)
def page_count(path: str, mtime: float) -> int:
    import pymupdf

    with pymupdf.open(path) as pdf:
        return pdf.page_count


@st.cache_data(show_spinner="Sayfa çiziliyor…", max_entries=24)
def page_png(path: str, mtime: float, page: int, zoom: float) -> bytes:
    return render_page_png(Path(path), page, zoom)


def md(text: str) -> str:
    """Escape user text so Streamlit markdown shows it literally."""
    return _MD_SPECIAL.sub(r"\\\1", text)


def flash(kind: str, message: str) -> None:
    ss["flash"] = (kind, message)


def show_flash() -> None:
    if "flash" in ss:
        kind, message = ss.pop("flash")
        getattr(st, kind)(message)


# --- PDF viewer --------------------------------------------------------------------


def set_view(doc: str, page: int | None) -> None:
    ss["view_doc"] = doc
    ss["view_page"] = page or 1


def shift_page(delta: int, last: int) -> None:
    ss["view_page"] = max(1, min(last, ss["view_page"] + delta))


def pdf_viewer(zoom: float) -> None:
    """Document selector, page navigation and the rendered page."""
    docs = doc_ids()
    if ss.get("view_doc") not in docs:
        ss["view_doc"] = docs[0]
    ss.setdefault("view_page", 1)
    doc = st.selectbox("Doküman", docs, key="view_doc")
    path = raw_path(doc)
    if not path.is_file():
        st.warning(f"PDF bulunamadı: `{path}`. Önce `scripts/download.py` ile indirin.")
        return
    try:
        mtime = path.stat().st_mtime
        last = page_count(str(path), mtime)
    except Exception as exc:
        st.error(f"PDF açılamadı: {exc}")
        return
    if not 1 <= ss["view_page"] <= last:
        st.warning(f"Sayfa {ss['view_page']} bu PDF'te yok (1–{last}); sınıra çekildi.")
        ss["view_page"] = max(1, min(last, ss["view_page"]))
    prev_col, page_col, next_col = st.columns([1, 2, 1], vertical_alignment="bottom")
    prev_col.button("← Önceki", key="prev_page", on_click=shift_page, args=(-1, last))
    page_col.number_input(
        f"Sayfa (PDF dizini, 1–{last})", min_value=1, max_value=last, step=1, key="view_page"
    )
    next_col.button("Sonraki →", key="next_page", on_click=shift_page, args=(1, last))
    try:
        st.image(page_png(str(path), mtime, ss["view_page"], zoom), width="stretch")
    except Exception as exc:
        st.error(f"Sayfa çizilemedi: {exc}")
        return
    st.caption("Okumak için görselin üstündeki tam ekran düğmesini kullanın.")


def evidence_selector(q: Question) -> None:
    if len(q.evidence) < 2:
        return
    key = f"evidence_choice_{q.id}"

    def jump() -> None:
        ev = q.evidence[ss[key]]
        set_view(ev.doc, ev.page)

    st.radio(
        "Kanıt",
        range(len(q.evidence)),
        format_func=lambda i: f"{q.evidence[i].doc} s.{q.evidence[i].page or '?'}",
        key=key,
        horizontal=True,
        on_change=jump,
    )


def default_view(q: Question) -> None:
    ss.pop(f"evidence_choice_{q.id}", None)  # selector starts again at the first entry
    if q.evidence:
        set_view(q.evidence[0].doc, q.evidence[0].page)
    elif q.banks and q.years and f"{q.banks[0]}-{q.years[0]}" in doc_ids():
        set_view(f"{q.banks[0]}-{q.years[0]}", 1)
    if q.evidence and q.evidence[0].page is None:
        flash("info", "Bu sorunun kanıt sayfası belirtilmemiş; sayfayı bulup düzenleyin.")


# --- question form (edit and add) --------------------------------------------------


def init_form(prefix: str, q: Question | None) -> None:
    exp = q.expected if q else None
    tol = exp.tolerance if exp else None
    values = {
        "question": q.question if q else "",
        "type": q.type if q else "olgusal",
        "banks": list(q.banks) if q else [],
        "years": ", ".join(map(str, q.years)) if q else "",
        "basis": (q.basis if q else None) or NONE,
        "behavior": q.expected_behavior if q else "answer",
        "answer": exp.answer if exp else "",
        "value": "" if exp is None or exp.value is None else str(exp.value),
        "unit": (exp.unit if exp else None) or NONE,
        "tol_type": tol.type if tol else NONE,
        "tol_amount": str(tol.amount) if tol else "",
        "evidence": format_evidence(q) if q else "",
        "key_points": "\n".join(q.key_points) if q else "",
        "notes": q.notes if q else "",
    }
    for name, value in values.items():
        ss[f"{prefix}_{name}"] = value


def set_evidence_from_view(prefix: str, append: bool) -> None:
    line = f"{ss['view_doc']} {ss['view_page']}"
    current = ss.get(f"{prefix}_evidence", "").strip()
    ss[f"{prefix}_evidence"] = f"{current}\n{line}" if append and current else line


def form_fields(prefix: str) -> None:
    def key(name: str) -> str:
        return f"{prefix}_{name}"

    st.text_area("Soru", key=key("question"), height=80)
    left, right = st.columns(2)
    left.selectbox("Tip", QUESTION_TYPES, key=key("type"))
    right.selectbox(
        "Beklenen davranış", BEHAVIORS, key=key("behavior"), format_func=BEHAVIOR_LABELS.get
    )
    left.multiselect("Bankalar", BANKS, key=key("banks"), placeholder="Seçin")
    right.text_input("Yıllar (virgülle)", key=key("years"))
    left.selectbox("Esas", BASES, key=key("basis"))
    st.text_area("Beklenen cevap", key=key("answer"), height=80)
    value, unit, tol_type, tol_amount = st.columns(4)
    value.text_input("Değer", key=key("value"), help="TL ve adet tam sayı; boş = sayısal değil")
    unit.selectbox("Birim", UNITS, key=key("unit"))
    tol_type.selectbox("Tolerans tipi", TOLERANCE_TYPES, key=key("tol_type"))
    tol_amount.text_input("Tolerans", key=key("tol_amount"), help="TL: rel 0.01 · %: abs 0.2")
    st.text_area(
        "Kanıt (her satıra `doküman sayfa`; bilinmeyen sayfa için `-`)",
        key=key("evidence"),
        height=80,
    )
    set_col, add_col = st.columns(2)
    set_col.button(
        "Kanıtı bu sayfaya ayarla",
        key=key("evidence_set"),
        on_click=set_evidence_from_view,
        args=(prefix, False),
    )
    add_col.button(
        "Bu sayfayı kanıta ekle",
        key=key("evidence_add"),
        on_click=set_evidence_from_view,
        args=(prefix, True),
    )
    st.text_area("Anahtar noktalar (sentez; her satıra bir madde)", key=key("key_points"))
    st.text_area("Notlar", key=key("notes"))


def collect_form(prefix: str) -> dict:
    """The editable fields from the form; raises ReviewError on unparsable input."""

    def get(name: str):
        value = ss[f"{prefix}_{name}"]
        return None if value == NONE else value

    tol_amount = parse_number(get("tol_amount"))
    if get("tol_type") is None:
        if tol_amount is not None:
            raise ReviewError("Tolerans miktarı girildi ama tolerans tipi seçilmedi.")
        tolerance = None
    else:
        if tol_amount is None:
            raise ReviewError("Tolerans tipi seçildi ama miktarı girilmedi.")
        tolerance = {"type": get("tol_type"), "amount": tol_amount}
    return {
        "question": get("question").strip(),
        "type": get("type"),
        "banks": list(get("banks")),
        "years": parse_years(get("years")),
        "basis": get("basis"),
        "expected_behavior": get("behavior"),
        "expected": {
            "answer": get("answer").strip(),
            "value": parse_number(get("value")),
            "unit": get("unit"),
            "tolerance": tolerance,
        },
        "evidence": parse_evidence(get("evidence")),
        "key_points": [line.strip() for line in get("key_points").splitlines() if line.strip()],
        "notes": get("notes").strip(),
    }


# --- actions (button callbacks; they run before the next script run) ---------------

STALE_MESSAGE = (
    "Bu soru siz baktıktan sonra dosyada değişmiş; hiçbir şey yazılmadı. "
    "Güncel hali yüklendi, kontrol edip tekrar deneyin."
)


def do_approve(path: Path, shown: Question) -> None:
    try:
        approve(path, shown)
        flash("success", f"{shown.id} onaylandı.")
    except StaleQuestionError:
        flash("error", STALE_MESSAGE)


def do_reject(path: Path, shown: Question) -> None:
    reason = ss.get(f"reject_reason_{shown.id}", "")
    try:
        reject(path, shown, reason)
        target = rejected_path_for(path).name
        flash("success", f"{shown.id} reddedildi ve {target} dosyasına taşındı.")
    except StaleQuestionError:
        flash("error", STALE_MESSAGE)
    except ReviewError as exc:
        flash("error", f"Reddedilemedi: {exc}")


def start_edit(q: Question) -> None:
    ss["editing"] = q.id
    ss["edit_base"] = q  # the row the edits apply to; a save refuses if disk differs
    ss.pop("form_error", None)
    init_form("edit", q)


def stop_edit() -> None:
    ss.pop("editing", None)
    ss.pop("edit_base", None)
    ss.pop("form_error", None)


def do_save(path: Path, and_approve: bool) -> None:
    base: Question = ss["edit_base"]
    try:
        edit(path, base, collect_form("edit"), approve=and_approve)
    except StaleQuestionError:
        ss["form_error"] = STALE_MESSAGE + " Düzenlemeyi iptal edip yeniden başlatın."
        return
    except ReviewError as exc:
        ss["form_error"] = f"Kaydedilmedi, değerler geçersiz: {exc}"
        return
    stop_edit()
    flash("success", f"{base.id} kaydedildi" + (" ve onaylandı." if and_approve else "."))


def do_add(path: Path) -> None:
    try:
        new = add_human_question(path, collect_form("add"))
    except ReviewError as exc:
        ss["add_error"] = f"Eklenmedi, değerler geçersiz: {exc}"
        return
    ss.pop("add_error", None)
    init_form("add", None)
    flash("success", f"{new.id} eklendi (split: {new.split}).")


def select_neighbour(ids: list[str], delta: int) -> None:
    i = ids.index(ss["current_qid"]) + delta
    ss["current_qid"] = ids[max(0, min(len(ids) - 1, i))]


# --- pages -------------------------------------------------------------------------


def show_question(q: Question) -> None:
    st.subheader(f"{q.id} · {q.type}")
    status = f"doğrulandı ({q.verified_at})" if q.verified else "doğrulanmadı"
    st.caption(
        f"split: {q.split} · pilot: {'evet' if q.pilot else 'hayır'} · yazar: {q.author} · "
        f"LLM kontrolü: {'evet' if q.llm_checked else 'hayır'} · durum: {status}"
    )
    exp = q.expected
    rows = [
        f"**Soru:** {md(q.question)}",
        f"**Bankalar:** {', '.join(q.banks) or '—'} · **Yıllar:** "
        f"{', '.join(map(str, q.years)) or '—'} · **Esas:** {q.basis or '—'}",
        f"**Beklenen davranış:** {BEHAVIOR_LABELS[q.expected_behavior]}",
        f"**Beklenen cevap:** {md(exp.answer)}",
    ]
    if exp.value is not None:
        tol = exp.tolerance
        tol_text = f"{tol.type} {tol.amount}" if tol else "—"
        rows.append(f"**Değer:** {exp.value:,} {md(exp.unit or '')} · **Tolerans:** {tol_text}")
    evidence = ", ".join(f"{e.doc} s.{e.page or '?'}" for e in q.evidence)
    rows.append(f"**Kanıt:** {evidence or '—'}")
    if q.key_points:
        rows.append("**Anahtar noktalar:**\n" + "\n".join(f"- {md(k)}" for k in q.key_points))
    if q.derived_from:
        rows.append(f"**Türetildiği sorular:** {', '.join(q.derived_from)}")
    if q.notes:
        rows.append(f"**Notlar:** {md(q.notes)}")
    st.markdown("\n\n".join(rows))


def review_actions(path: Path, q: Question) -> None:
    approve_col, edit_col = st.columns(2)
    approve_col.button(
        "Onayla",
        key=f"approve_{q.id}",
        type="primary",
        disabled=q.verified,
        on_click=do_approve,
        args=(path, q),
        help="Soruyu ve cevabı PDF sayfasında kontrol ettiyseniz.",
    )
    edit_col.button("Düzenle", key=f"edit_{q.id}", on_click=start_edit, args=(q,))
    with st.expander("Reddet"):
        st.text_input("Gerekçe (isteğe bağlı)", key=f"reject_reason_{q.id}")
        st.button("Reddetmeyi onayla", key=f"reject_{q.id}", on_click=do_reject, args=(path, q))


def edit_panel(path: Path, q: Question) -> None:
    st.subheader(f"{q.id} düzenleniyor")
    if ss["edit_base"] != q:
        st.warning("Bu soru düzenlemeye başladıktan sonra dosyada değişti; kayıt reddedilecek.")
    form_fields("edit")
    if "form_error" in ss:
        st.error(ss["form_error"])
    save_col, save_approve_col, cancel_col = st.columns(3)
    save_col.button("Kaydet", key="save", on_click=do_save, args=(path, False))
    save_approve_col.button(
        "Kaydet ve onayla", key="save_approve", type="primary", on_click=do_save, args=(path, True)
    )
    cancel_col.button("Vazgeç", key="cancel_edit", on_click=stop_edit)
    st.caption("Kaydetmek doğrulama durumunu değiştirmez; onay için “Kaydet ve onayla”.")


def matches(q: Question, filters: dict) -> bool:
    if not filters["show_verified"] and q.verified:
        return False
    if filters["types"] and q.type not in filters["types"]:
        return False
    if filters["splits"] and q.split not in filters["splits"]:
        return False
    if filters["authors"] and q.author not in filters["authors"]:
        return False
    return not filters["banks"] or any(b in filters["banks"] for b in q.banks)


def review_page(path: Path, questions: list[Question], zoom: float) -> None:
    if not questions:
        st.info(
            f"Henüz soru yok: `{path}` bulunamadı ya da boş. Taslak sorular üretildiğinde "
            "burada listelenir. Kenar çubuğundan **Yeni soru ekle** moduna geçip kendi "
            "sorularınızı ekleyebilirsiniz."
        )
        return
    with st.sidebar:
        st.markdown("**Filtreler**")
        filters = {
            "types": st.multiselect("Tip", QUESTION_TYPES, key="filter_types", placeholder="Hepsi"),
            "splits": st.multiselect("Split", SPLITS, key="filter_splits", placeholder="Hepsi"),
            "banks": st.multiselect("Banka", BANKS, key="filter_banks", placeholder="Hepsi"),
            "authors": st.multiselect("Yazar", AUTHORS, key="filter_authors", placeholder="Hepsi"),
            "show_verified": st.toggle("Doğrulanmışları da göster", key="show_verified"),
        }
    shown = [q for q in questions if matches(q, filters)]
    if not shown:
        st.success("Filtrelere uyan doğrulanmamış soru kalmadı.")
        return
    ids = [q.id for q in shown]
    current = ss.get("current_qid")
    if current not in ids:
        # after approve/reject, move on to the next question of the previous list
        previous = ss.get("last_ids", [])
        after = previous[previous.index(current) + 1 :] if current in previous else []
        ss["current_qid"] = next((i for i in after if i in ids), ids[0])
    ss["last_ids"] = ids
    by_id = {q.id: q for q in shown}
    with st.sidebar:
        st.selectbox(
            f"Soru ({len(ids)} tane)",
            ids,
            key="current_qid",
            format_func=lambda i: (
                f"{i} · {by_id[i].question[:50]}" + (" [onaylı]" if by_id[i].verified else "")
            ),
        )
        prev_col, next_col = st.columns(2)
        prev_col.button("← Önceki soru", key="prev_q", on_click=select_neighbour, args=(ids, -1))
        next_col.button("Sonraki soru →", key="next_q", on_click=select_neighbour, args=(ids, 1))
    q = by_id[ss["current_qid"]]
    if ss.get("view_for") != q.id:
        ss["view_for"] = q.id
        if ss.get("editing") != q.id:
            stop_edit()
        default_view(q)
        show_flash()
    left, right = st.columns([2, 3])
    with left:
        if ss.get("editing") == q.id:
            edit_panel(path, q)
        else:
            show_question(q)
            review_actions(path, q)
    with right:
        evidence_selector(q)
        pdf_viewer(zoom)


def add_page(path: Path, zoom: float) -> None:
    if "add_question" not in ss:
        init_form("add", None)
    left, right = st.columns([2, 3])
    with left:
        st.subheader("Yeni soru ekle")
        st.info(ADD_HELP)
        form_fields("add")
        if "add_error" in ss:
            st.error(ss["add_error"])
        st.button("Soruyu ekle", key="add_submit", type="primary", on_click=do_add, args=(path,))
    with right:
        pdf_viewer(zoom)


def sidebar_progress(path: Path, questions: list[Question]) -> None:
    p = progress(questions)
    done, total = p["total"]
    st.metric("Doğrulanan", f"{done} / {total}")
    lines = [f"- {t}: {d} / {n}" for t, (d, n) in p["by_type"].items() if n]
    if lines:
        st.markdown("\n".join(lines))
    st.caption(f"Dosya: `{path}`")


def main() -> None:
    st.set_page_config(page_title="kaprag gold set inceleme", layout="wide")
    path = questions_path()
    try:
        questions = load_for_review(path)
    except (GoldSetError, OSError) as exc:
        st.error(f"Soru dosyası okunamadı: {exc}")
        st.stop()
    with st.sidebar:
        st.title("Gold set inceleme")
        mode = st.radio("Mod", ["İnceleme", "Yeni soru ekle"], key="mode", horizontal=True)
        zoom = st.slider("Çözünürlük", 1.0, 3.0, 2.0, 0.5, key="zoom")
        sidebar_progress(path, questions)
    show_flash()
    if mode == "İnceleme":
        review_page(path, questions, zoom)
    else:
        add_page(path, zoom)


if __name__ == "__main__":
    main()
