"""Chat demo: ask the reports questions, see cited pages.

Run (needs the `app` dependency group: `uv sync --group ml --group app --group observability`):
  uv run streamlit run scripts/chat_app.py
  uv run streamlit run scripts/chat_app.py -- --config rerank

The pipeline runs in this process through `kaprag.serve` (the ask API need not run).
Its config is `--config` (after `--`), else $KAPRAG_SERVE_CONFIG, else `router`; the
LLM backend is LLM_BACKEND (environment or `.env`). $KAPRAG_SERVE_FACTORY
(`module:function`, default `kaprag.serve:open_service`) replaces the service factory,
e.g. with a fake in tests. Each question is answered on its own; the conversation is
kept for display only. The repo's `.streamlit/config.toml` binds to 127.0.0.1.
"""

from __future__ import annotations

import argparse
import importlib
import os
import re
import sys
from typing import Any

import streamlit as st
from dotenv import load_dotenv

from kaprag.access.roles import ROLES
from kaprag.corpus import REPO_ROOT
from kaprag.serve import (
    CONFIG_ENV,
    AskResult,
    InvalidRequestError,
    PageNotFoundError,
    PipelineFailedError,
    ServiceClosedError,
)

FACTORY_ENV = "KAPRAG_SERVE_FACTORY"
DEFAULT_FACTORY = "kaprag.serve:open_service"
ROLE_LABELS = {
    "public": "public — herkese açık içerik",
    "analyst": "analyst — + finansal tablo dipnotları",
    "risk": "risk — + risk yönetimi içeriği",
}
BACKEND_LABELS = {
    "claude_code": "claude_code (Claude aboneliği; yalnızca yerel kullanım)",
    "anthropic_api": "anthropic_api (API anahtarı)",
}
ROUTE_LABELS = {
    "simple": "basit: tek arama + tek cevap",
    "multi": "çok adımlı: soru parçalandı, ayrı aramalar, hesap ve birleştirme",
}
PAGE_ERRORS = {
    "unknown_doc": "Bu doküman manifestte yok.",
    "page_range": "Bu sayfa dokümanda yok.",
    "missing_pdf": "PDF bu bilgisayarda yok (data/raw/); önce `scripts/download.py` ile indirin.",
}
SIMULATED_NOTE = (
    "Erişim modeli simüledir: raporlar kamuya açıktır, roller yalnızca erişim "
    "filtrelemesini göstermek için kurala göre atanmıştır."
)
_MD_SPECIAL = re.compile(r"([\\`*_\[\]#<>|~$])")
ss = st.session_state


def md(text: str) -> str:
    """Escape model or user text so Streamlit markdown shows it literally."""
    return _MD_SPECIAL.sub(r"\\\1", text)


def config_name() -> str | None:
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--config")
    args, _ = parser.parse_known_args(sys.argv[1:])
    return args.config or os.environ.get(CONFIG_ENV) or None


@st.cache_resource(show_spinner="Modeller yükleniyor (ilk açılışta biraz sürer)…")
def get_service(factory_ref: str, config: str | None) -> Any:
    """One service per process: the models load once and stay loaded until the process
    exits (Streamlit has no shutdown hook; exiting releases the GPU and the connection)."""
    load_dotenv(REPO_ROOT / ".env", override=False)
    module_name, _, attr = factory_ref.partition(":")
    factory = getattr(importlib.import_module(module_name), attr)
    return factory(config)


# --- one answer --------------------------------------------------------------------


def number(value: int | float) -> str:
    """Turkish digit grouping: 1.234.567,8."""
    text = f"{value:,}" if isinstance(value, int) else f"{value:,.2f}".rstrip("0").rstrip(".")
    return text.replace(",", "\0").replace(".", ",").replace("\0", ".")


def toggle_page(key: str, target: tuple[str, int]) -> None:
    ss[key] = None if ss.get(key) == target else target


def show_page(service: Any, doc: str, page: int) -> None:
    try:
        png = service.page_png(doc, page)
    except PageNotFoundError as exc:
        st.warning(f"{doc} s.{page} gösterilemiyor: {PAGE_ERRORS[exc.reason]}")
        return
    except Exception as exc:
        st.error(f"Sayfa çizilemedi: {type(exc).__name__}")
        return
    st.image(png, caption=f"{doc} · PDF sayfası {page}", width="stretch")


def show_citations(service: Any, result: AskResult, index: int) -> None:
    if not result.citations:
        return
    st.markdown("**Kaynaklar** (sayfayı görmek için tıklayın)")
    key = f"open_page_{index}"
    columns = st.columns(min(len(result.citations), 4))
    for i, cite in enumerate(result.citations):
        name = f"{cite.bank_name} {cite.year}" if cite.bank_name else cite.doc
        columns[i % len(columns)].button(
            f"{name} · s.{cite.page}",
            key=f"cite_{index}_{i}",
            on_click=toggle_page,
            args=(key, (cite.doc, cite.page)),
        )
    if ss.get(key):
        doc, page = ss[key]
        with st.expander(f"{doc} · sayfa {page}", expanded=True):
            show_page(service, doc, page)


def show_how(result: AskResult) -> None:
    with st.expander("Bu cevap nasıl üretildi?"):
        route = ROUTE_LABELS.get(result.route or "", result.route or "bilinmiyor")
        cached = f" ({result.cached_llm_calls} önbellekten)" if result.cached_llm_calls else ""
        st.markdown(
            f"- **Yol:** {route}\n"
            f"- **Rol:** {result.role}\n"
            f"- **Getirilen kaynak parçası:** {len(result.sources)} · "
            f"**atıf:** {len(result.citations)}\n"
            f"- **Süre:** {result.latency_ms / 1000:.1f} sn\n"
            f"- **LLM çağrısı:** {result.llm_calls}{cached} · "
            f"**API eşdeğeri maliyet:** ${result.cost_usd:.4f}\n"
            f"- **Yapılandırma:** {result.config} · **LLM arka ucu:** {result.backend}"
        )
        if result.sources:
            st.markdown(
                "\n".join(
                    f"{s.rank}. {s.doc} s.{s.page_start}"
                    + (f"–{s.page_end}" if s.page_end != s.page_start else "")
                    + f" (skor {s.score:.3f})"
                    for s in result.sources
                )
            )


def show_answer(service: Any, result: AskResult, index: int) -> None:
    if result.abstained:
        st.warning("**Belgelerde cevap bulunamadı.** " + md(result.answer))
    elif result.clarification:
        st.info("**Netleştirme gerekiyor:** " + md(result.clarification))
        if result.answer:
            st.markdown(md(result.answer))
    else:
        st.markdown(md(result.answer))
        facts = []
        if result.value is not None:
            facts.append(f"**Değer:** {number(result.value)} {md(result.unit or '')}".rstrip())
        if result.basis:
            facts.append(f"**Esas:** {result.basis}")
        if facts:
            st.markdown(" · ".join(facts))
    show_citations(service, result, index)
    show_how(result)


# --- page --------------------------------------------------------------------------


def clear_chat() -> None:
    ss["messages"] = []
    for key in [k for k in ss if str(k).startswith("open_page_")]:
        del ss[key]


def sidebar(service: Any) -> str:
    with st.sidebar:
        st.title("kaprag")
        st.caption("Akbank, Garanti BBVA, İş Bankası ve Yapı Kredi 2023–2025 faaliyet raporları")
        role = st.selectbox("Rol", ROLES, key="role", format_func=ROLE_LABELS.get)
        st.caption(SIMULATED_NOTE)
        backend = BACKEND_LABELS.get(service.backend, service.backend)
        st.markdown(f"**LLM arka ucu:** {backend}  \n**Yapılandırma:** {service.config}")
        st.caption(
            "Her soru bağımsız cevaplanır; önceki mesajlar modele gönderilmez, sohbet "
            "yalnızca görüntülemek içindir."
        )
        st.button("Sohbeti temizle", key="clear", on_click=clear_chat)
    return role


def ask(service: Any, question: str, role: str) -> dict[str, Any]:
    try:
        with st.spinner("Cevap aranıyor…"):
            return {"question": question, "result": service.ask(question, role=role)}
    except InvalidRequestError as exc:
        return {"question": question, "error": f"Soru geçersiz: {exc}"}
    except PipelineFailedError:
        return {"question": question, "error": "Cevap üretilemedi; bir hata oluştu."}
    except ServiceClosedError:
        return {"question": question, "error": "Servis kapandı; sayfayı yenileyin."}


def main() -> None:
    st.set_page_config(page_title="kaprag sohbet", layout="wide")
    factory_ref = os.environ.get(FACTORY_ENV) or DEFAULT_FACTORY
    try:
        service = get_service(factory_ref, config_name())
    except Exception as exc:
        st.error(f"Pipeline açılamadı: {type(exc).__name__}: {exc}")
        st.stop()
    role = sidebar(service)
    ss.setdefault("messages", [])
    st.title("Faaliyet raporlarına sorun")
    if not ss["messages"]:
        st.info(
            "Örnek: “Akbank 2024 yıl sonunda kaç şubeyle hizmet veriyordu?” ya da "
            "“2023'te Garanti BBVA'nın konsolide net kârı Yapı Kredi'ninkinden yüzde kaç "
            "fazlaydı?” Soruları İngilizce de sorabilirsiniz."
        )
    question = st.chat_input("Sorunuzu yazın", key="question")
    if question:
        ss["messages"].append(ask(service, question, role))
    for index, message in enumerate(ss["messages"]):
        with st.chat_message("user"):
            st.markdown(md(message["question"]))
        with st.chat_message("assistant"):
            if "error" in message:
                st.error(message["error"])
            else:
                show_answer(service, message["result"], index)


if __name__ == "__main__":
    main()
