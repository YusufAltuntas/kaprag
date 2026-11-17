# CLAUDE.md — kaprag

> Bu dosya bu repodaki çalışma kurallarının ve doküman haritasının sahibidir.

## Doküman haritası

- `PLAN.md` — kapsam, mimari kararlar, görev listesi ve durumu, karar günlüğü, engeller, sabah raporu.
- `eval/README.md` — eval seti tasarımı: soru tipleri, şema, puanlama kuralları, metrikler, gold set üretimi.
- `README.md` (Türkçe) / `README.en.md` (İngilizce) — dışarıya dönük proje tanıtımı. Diğer dokümanlardaki bilgiyi kopyalamaz; özetler ve link verir.

Bir bilgi tek dokümanda yaşar. Yeni doküman açmadan önce bunlardan biri sahiplenebiliyor mu bak.

## Dil (bu repo için global kuraldan farklı)

- Kod tanımlayıcıları, **kod yorumları, docstring'ler ve commit mesajları İngilizce.**
- `README.md` Türkçe, `README.en.md` İngilizce. İç dokümanlar (`PLAN.md`, `eval/README.md`, bu dosya) Türkçe.
- Arayüz metinleri Türkçe; sistemin cevapları sorunun dilinde.

## Kurallar

- Paket yöneticisi `uv`; komutlar `uv run …`. Lint: `uv run ruff check .` · Test: `uv run pytest`.
- Kod hem Windows hem macOS'ta çalışır: yollar `pathlib` ile, sabit `/` ayırıcı yok, shell'e özgü komut yok.
- LLM çağrıları yalnızca `kaprag.llm` üzerinden yapılır; başka yerde doğrudan SDK çağrısı olmaz.
- Sırlar `.env`'de, `.env` git'e girmez. `LLM_BACKEND=claude_code` iken `ANTHROPIC_API_KEY` tanımlı olmamalı (tanımlıysa abonelik yerine API faturalanır).
- Veri: `data/manifest.csv` git'te; `data/raw/` (PDF) ve `data/parsed/` git dışı.
- Deney çıktıları `runs/YYYY-MM-DD-<konu>/` altında (git dışı). Her run klasöründe ne test ettiğini tek satırda söyleyen bir `README.md` bulunur.
- Gold set `eval/gold/questions.jsonl` git'te. `verified: true` yalnızca kullanıcı tarafından review arayüzünden verilir.
- Test split'i yalnızca önemli versiyonlarda çalıştırılır (bkz. `eval/README.md` §2).
