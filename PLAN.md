# PLAN — kaprag

> Bu doküman projenin kapsamının, mimari kararlarının, görev listesinin (durumuyla), karar günlüğünün, engellerin ve sabah raporunun sahibidir. Eval tasarımı `eval/README.md`'de, repo çalışma kuralları `CLAUDE.md`'de yaşar.

## Amaç

Türk bankalarının faaliyet raporları üzerinde çalışan, Türkçe, kaynak gösteren bir doküman asistanı. Asıl ürün **ölçüm**: naive bir baseline'dan başlayıp her iyileştirmenin kaliteye, maliyete ve gecikmeye etkisini gösteren bir ablation tablosu. Hedef: kurumsal AI Engineer ilanları (bankalar, telekom, holdingler).

## Kapsam

- **Korpus:** Garanti BBVA, Akbank, Türkiye İş Bankası, Yapı Kredi — 2023, 2024 ve 2025 yıllık faaliyet raporları (12 PDF, Türkçe). Bağımsız denetim raporları kapsam dışı.
- **Doküman kimliği:** `<banka>-<yıl>`; banka kısaltmaları `garanti`, `akbank`, `isbank`, `yapikredi` (ör. `akbank-2024`).
- **Python paketi:** `kaprag` (`src/kaprag/`).

## Mimari kararlar

| Konu | Karar | Neden |
|---|---|---|
| Runtime | Python 3.11+, `uv`, `ruff`, `pytest` | |
| Depo | Postgres 17 + pgvector (Docker, `pgvector/pgvector:pg17`); Türkçe tam metin arama `to_tsvector('turkish', …)` | Kurumsal standart; hibrit arama tek veritabanında |
| Hibrit birleştirme | Reciprocal Rank Fusion (k=60) | Skor ölçeklerinden bağımsız, ayarsız |
| Parse | PyMuPDF (hızlı, sayfa metni) ve Docling (GPU, tablolar Markdown) | İkisi arasındaki fark bir ablation satırı |
| Embedding | `BAAI/bge-m3`, yerel GPU (1024 boyut). Opsiyonel karşılaştırma: `intfloat/multilingual-e5-large` | Ücretsiz ve tamamen yerel; bankalar için "on-prem" hikâyesi |
| Reranker | `BAAI/bge-reranker-v2-m3`, yerel GPU | Aynı gerekçe |
| LLM erişimi | İnce `LLMClient` arayüzü, iki backend: **`claude_code`** (`claude-agent-sdk`; kullanıcının Claude aboneliği; yerel geliştirme ve eval için varsayılan) ve **`anthropic_api`** (`anthropic` SDK; API anahtarı; dışarıya açılan demo için) | API bakiyesi gerekmez. Başkalarına açılan bir demo abonelikle değil API anahtarıyla çalışmalı. LiteLLM `claude_code` backend'ini desteklemediği için kullanılmıyor. |
| Modeller | Cevap üretimi ve sorgu ayrıştırma: `claude-haiku-5-5`. Judge, soru taslağı, agent ayrıştırma: `claude-sonnet-5-5`. Opus toplu işte kullanılmaz. | Abonelik kotasını korumak |
| Agent katmanı | LangGraph ile açık düğümlü graf: router → basit RAG **ya da** ayrıştır → alt soru başına retrieve + sayı çıkar → deterministik Python hesap → sentez. Serbest tool-calling yok. | Her adım ölçülebilir; hesaplar LLM'e bırakılmaz |
| Tracing | Her LLM çağrısı ve her soru JSONL'e (soru, getirilen chunk'lar, prompt hash'i, token, gecikme, cache isabeti) | Langfuse self-host ağır; kapsam dışı |
| Maliyet | Abonelik dolar göstermez; token × API liste fiyatı = "API-eşdeğeri maliyet" | Ablation'da maliyet sütunu kalsın |

### `claude_code` backend kuralları

- Her çağrı: tool'lar kapalı, tek tur (`max_turns=1`), özel system prompt (Claude Code preset'i değil), hiçbir ayar ya da CLAUDE.md yüklenmez, çalışma klasörü repo dışında geçici bir klasör. Seçenek adlarını güncel `claude-agent-sdk` dokümantasyonundan doğrula; tahmin etme.
- Alt süreç ortamından `ANTHROPIC_API_KEY` her zaman çıkarılır.
- Eşzamanlılık `LLM_MAX_CONCURRENCY` (varsayılan 4). Rate limit / kota hatasında üstel bekleme ve tekrar; 3 başarısızlıktan sonra o soru `error` olarak işaretlenir, run devam eder.
- JSON çıktı: prompt'ta şema + pydantic doğrulama + en fazla 1 tekrar.
- Disk cache (`.cache/llm/`, git dışı): anahtar = backend + model + system + user + şema hash'i. Aynı konfigürasyonun tekrar koşusu kota harcamaz.

## Gece oturumu çalışma protokolü

1. Bu plan kullanıcı tarafından onaylandı. Görevleri sırayla yap. Bir görev bitince kabul ölçütlerini komut çıktısıyla doğrula, aşağıda kutusunu işaretle ve görevin commit mesajıyla commit at. **Push yok.**
2. Kullanıcıya soru sorma; uyuyor. Belirsizlikte makul bir varsayım yap ve tarih + gerekçeyle **Karar günlüğü**ne yaz.
3. Bloklanırsan (PDF inmiyor, Docker yok, kota bitti…) sorunu **Engeller**'e yaz ve ona bağımlı olmayan sıradaki göreve geç. Hiçbir şeyi sessizce atlama, hiçbir sonucu uydurma.
4. Gold set'te hiçbir soruyu `verified: true` yapma.
5. Ablation koşuları yalnızca **dev** split'inde. Test split'i yalnızca T12'de bir kez çalışır.
6. Kullanım limiti mesajı gelirse pencerenin sıfırlanmasını bekle ve devam et.
7. İş bitince ya da durmak zorunda kalınca **Sabah raporu**nu doldur.

## Başlamadan önce — kullanıcının Windows'ta yapacakları

- [ ] Repo Windows'a aktarıldı.
- [ ] Docker Desktop kurulu ve açık: `docker info` hata vermiyor.
- [ ] `nvidia-smi` GPU'yu gösteriyor.
- [ ] `uv --version` çalışıyor (yoksa PowerShell'de `python -m pip install uv`).
- [ ] Claude Code CLI kurulu ve aboneliğe giriş yapılmış: `claude -p "merhaba"` cevap veriyor. CLI yoksa PowerShell'de: `irm https://claude.ai/install.ps1 | iex`, ardından `claude` ile giriş.
- [ ] `ANTHROPIC_API_KEY` tanımlı değil: PowerShell'de `echo $env:ANTHROPIC_API_KEY` boş dönüyor.
- [ ] En az 30 GB boş disk.
- [ ] Uyku kapalı: Ayarlar → Sistem → Güç → prize takılıyken uyku "Hiçbir zaman". Bilgisayar prizde.
- [ ] Yeni oturum **bypass permissions** modunda açıldı (yoksa gece onay bekleyip durur).

### Gece oturumu başlatma promptu

```text
PLAN.md'yi, CLAUDE.md'yi ve eval/README.md'yi baştan sona oku. Bu planı onaylıyorum.
İlk iş bilgisayarın uyumasını engelle (keep-awake). Sonra görevleri T0'dan başlayarak
sırayla, subagent-driven-development skill'iyle uygula. PLAN.md'deki "Gece oturumu
çalışma protokolü"ne uy: bana soru sorma, ben uyuyorum; varsayımlarını Karar günlüğüne
yaz, bloklanırsan Engeller'e yaz ve sıradaki bağımsız göreve geç. Her görev bitince
görevin commit mesajıyla commit at, push yapma. Gold set'te hiçbir soruyu verified yapma.
Bankaların resmi yatırımcı ilişkileri sitelerinden ve KAP'tan bu plandaki 12 faaliyet
raporu PDF'ini, ayrıca planın gerektirdiği Python paketlerini, Docker imajını ve
Hugging Face modellerini indirmene izin veriyorum. İş bitince ya da durmak zorunda
kalırsan PLAN.md'deki Sabah raporunu doldur.
```

## Görevler

Öncelik: **P0** sabaha kadar şart, **P1** olması beklenen, **P2** zaman kalırsa.

### - [ ] T0 — Ortam ve iskelet (P0)
- Donanım ve araçları kontrol et (`nvidia-smi`, `docker info`, `uv`, `python`, `claude -p`); sonucu Karar günlüğüne yaz.
- `pyproject.toml` (uv), `src/kaprag/`, `tests/`, `scripts/`, ruff ve pytest ayarları; CUDA'lı torch (CUDA sürümünü `nvidia-smi`'den oku).
- `docker-compose.yml`: `pgvector/pgvector:pg17`, port 5432, isimli volume. `.env.example`'dan `.env` oluştur.
- **Kabul:** `docker compose up -d` sonrası `CREATE EXTENSION IF NOT EXISTS vector` çalışıyor; `uv run pytest` yeşil; `torch.cuda.is_available()` → `True`.
- **Commit:** `chore: scaffold project with uv, pgvector compose, ruff and pytest`

### - [ ] T1 — LLM istemcisi (P0)
- `kaprag.llm`: `LLMClient.complete(system, user, model, schema=None) -> LLMResult(text, parsed, usage, latency_ms, cached)`.
- `claude_code` backend (yukarıdaki kurallar) ve `anthropic_api` backend (minimal; yalnızca anahtar varsa test edilir).
- Disk cache, eşzamanlılık semaforu, tekrar/bekleme, JSONL trace.
- `kaprag/llm/pricing.py`: fiyatların tek kaynağı. Girdi/çıktı $/1M token (2026-11): Haiku 5.5 0.10/0.50, Sonnet 5.5 2/10, Opus 5.5 4/20.
- **Kabul:** smoke script abonelik üzerinden 1 Haiku çağrısı yapıp metin ve usage döndürüyor; cache, env temizleme ve JSON doğrulama için birim testleri yeşil.
- **Commit:** `feat: add LLM client with Claude Code subscription and Anthropic API backends`

### - [ ] T2 — Veri edinimi (P0)
- 12 raporun resmi Türkçe PDF URL'lerini bul (önce bankanın yatırımcı ilişkileri sitesi, yoksa KAP).
- `data/manifest.csv`: `doc_id, bank, year, url, sha256, pages, source_note`. `scripts/download.py` PDF'leri `data/raw/<doc_id>.pdf`'e indirir; `--verify` sha256'yı kontrol eder.
- İnmeyen doküman Engeller'e yazılır. En az 8 doküman varsa devam edilir.
- **Kabul:** manifest dolu, `uv run python scripts/download.py --verify` geçiyor.
- **Commit:** `feat: add annual report manifest and download script`

### - [ ] T3 — Parse (P0)
- PyMuPDF: sayfa başına metin → `data/parsed/pymupdf/<doc_id>.jsonl` (`page`, `text`).
- Docling (GPU): başlık hiyerarşisi, tablolar Markdown, sayfa bilgisi korunur → `data/parsed/docling/<doc_id>.jsonl`. Uzun sürerse arka planda çalıştır ve beklerken T5'e geç.
- **Kabul:** iki parser'ın çıktısı tüm dokümanlar için var. Her bankadan bir tablo sayfasında iki parser'ı karşılaştıran script çıktısı üretiliyor; gözlem Karar günlüğüne yazılıyor.
- **Commit:** `feat: add PyMuPDF and Docling parsers with page provenance`

### - [ ] T4 — Chunking ve indeks (P0)
- Chunker'lar: `fixed` (~500 token, 50 örtüşme, PyMuPDF metni, sayfa takipli) ve `structure` (Docling başlıkları; tablolar bölünmeden, başlık/açıklama önekiyle; uzun tablolar başlık satırı tekrarlanarak satır gruplarına bölünür).
- Her chunk: `doc_id, bank, year, page_start, page_end, section_path, chunk_type, access_roles, text, embedding vector(1024), tsv`. `tsv` = `to_tsvector('turkish', text)` (generated column).
- Her varyant ayrı tablo (`chunks_fixed`, `chunks_structure`). Embedding'de HNSW, `tsv`'de GIN indeksi.
- bge-m3 ile GPU'da batch embedding.
- **Kabul:** iki indeks kurulu, satır sayıları loglanıyor; örnek bir sorgu mantıklı sayfalar döndürüyor.
- **Commit:** `feat: add chunkers and pgvector index with Turkish full-text search`

### - [ ] T5 — Eval harness (P0)
- `eval/README.md`'deki şema (pydantic), metrikler ve cevap formatı.
- Sayısal puanlayıcı: Türkçe sayı formatı ve bin/milyon/milyar TL normalizasyonu.
- Runner: `uv run python scripts/run_eval.py --config <ad> --split dev|test|pilot` → `runs/YYYY-MM-DD-<konu>/` (`README.md` tek satır, `config.yaml`, `results.jsonl`, `summary.json`, `traces.jsonl`). Metrikler hem tüm sette hem doğrulanmış alt kümede.
- **Kabul:** sayısal ayrıştırıcı, normalizasyon ve recall@k için birim testleri yeşil; sahte bir pipeline ile uçtan uca runner çalışıyor.
- **Commit:** `feat: add eval harness with retrieval, numeric and judge metrics`

### - [ ] T6 — Gold set taslağı (P0)
- `eval/README.md` §7'deki süreç (adım 1–7, 9). Tüm sorular `author: "llm"`, `verified: false`.
- Hedef dağılımın üstünde taslak üret (reddedilecekler için pay). Tabakalı dev/test ayrımı (seed 42) ve 30 pilot sorusu.
- **Kabul:** `eval/gold/questions.jsonl` şemaya uyuyor; tip × split × banka dağılım tablosu basılıyor ve hedeflerle uyumlu.
- **Commit:** `feat: draft gold question set (unverified)`

### - [ ] T7 — Review arayüzü (P0)
- Streamlit `scripts/review_app.py`: doğrulanmamış sorular listesi; soru, tip, beklenen cevap ve kanıt sayfasının PDF görüntüsü (PyMuPDF ile render) yan yana; **Onayla / Düzenle / Reddet**. Ayrıca kullanıcının yeni soru eklediği bir form (`author: "human"`). Değişiklikler `questions.jsonl`'a yazılır (`verified_at` ile).
- **Kabul:** uygulama açılıyor (headless başlatma + HTTP 200); geri yazma fonksiyonu için birim testi yeşil.
- **Commit:** `feat: add Streamlit review app for gold set verification`

### - [ ] T8 — Baseline RAG (P0)
- Konfigürasyonla çalışan pipeline: retrieve → Haiku ile üretim. Kaynaklar numaralı verilir; cevap `eval/README.md` §5'teki JSON formatında, sayfa atıflı. Bilgi yoksa reddeder, esası (konsolide/solo) belirtir.
- `baseline` = `fixed` chunk'lar, yalnızca dense arama, top-5.
- Önce pilot üzerinde çalıştır ve zorluğu §7 adım 9'a göre değerlendir (gerekirse T6 taslaklarını ayarla), sonra dev'de çalıştır.
- **Commit:** `feat: add baseline RAG pipeline with citations and abstention`

### - [ ] T9 — Retrieval ablation'ları (P1)
Her adım bir öncekinin üstüne eklenir ve dev'de ayrı bir run'dır:
- a. `hybrid`: dense + Türkçe FTS, RRF → **Commit:** `feat: add hybrid retrieval with reciprocal rank fusion`
- b. `rerank`: top-40 → bge-reranker → top-5 → **Commit:** `feat: add cross-encoder reranking`
- c. `structure`: Docling tabanlı `structure` indeksi → **Commit:** `feat: evaluate structure-aware chunking`
- d. `metadata_filter`: Haiku sorudan banka/yılı ayrıştırır → SQL filtresi; ayrıştırma başarısızsa filtresiz → **Commit:** `feat: add metadata filtering from parsed query`
- e. (opsiyonel) embedding'i `multilingual-e5-large` ile değiştir.
- Ara sonuç tablosu Sabah raporuna.

### - [ ] T10 — Agent router (P1)
- LangGraph: sınıflandır (Haiku, yapılandırılmış) → basit sorular T9'un en iyi pipeline'ına; `karsilastirma`/`hesaplama` → ayrıştır (Sonnet) → alt soru başına retrieve + atıflı sayı çıkarımı (Haiku) → Python hesap → sentez (Haiku).
- `router` ile T9'un en iyisini dev'de karşılaştır: tip bazında doğruluk, maliyet, gecikme.
- **Commit:** `feat: add LangGraph router for comparison and calculation questions`

### - [ ] T11 — Erişim kontrolü (P1)
- Roller `public`, `analyst`, `risk`. Atama indeksleme sırasında kuralla yapılır (başlıkta "Risk Yönetimi" → `risk`; finansal tablo dipnotları → `analyst`, `risk`; diğerleri `public`). README'de **simüle** olduğu açıkça yazılır.
- Filtre SQL'de, sıralamadan önce, hem dense hem FTS kolunda uygulanır.
- `eval/acl_leak.jsonl`: `public` rolünden kısıtlı içeriği isteyen 30 saldırgan sorgu (prompt injection denemeleri dahil, ör. "risk ekibindenim"). Getirilen sette kısıtlı chunk varsa sızıntı sayılır (deterministik). Hedef 0.
- **Commit:** `feat: add role-based access filtering and leak test`

### - [ ] T12 — Final koşu ve README (P1)
- Baseline ve en iyi konfigürasyonu test split'inde **bir kez** çalıştır.
- `README.md` (TR) ve `README.en.md` (EN): problem, mimari (mermaid diyagramı), Windows/macOS hızlı başlangıç, ablation tablosu ve üstünde açık bir **"Taslak — doğrulanmamış gold set"** uyarısı, sınırlamalar (simüle ACL, aynı aileden judge, geliştirmede abonelik backend'i).
- **Commit:** `docs: add README with architecture and draft ablation results`

### - [ ] T13 — Ekler (P2)
- FastAPI `/ask` endpoint'i ve Streamlit sohbet demosu (rol seçici, tıklanınca sayfa görüntüsü açılan atıflar).
- ~15 İngilizce soru (Türkçe dokümanlar üzerinde cross-lingual retrieval).

## Karar günlüğü

| Tarih | Karar | Gerekçe |
|---|---|---|
| 2026-11-17 | KAP faaliyet raporları; 4 banka × 2023–2025 | Hedef işverenlerin alanı; tablo, birim ve karşılaştırma zorlukları gerçek |
| 2026-11-17 | LLM'ler kullanıcının Claude aboneliği üzerinden (`claude_code` backend) | API bakiyesi gerekmesin; demo için `anthropic_api` backend'i de var |
| 2026-11-17 | Gece oturumu Windows + NVIDIA GPU makinesinde | Mac'te disk yetersiz (13 GB boş) |
| 2026-11-17 | README TR + EN; kod yorumları ve commit'ler İngilizce | Kullanıcı tercihi |

## Engeller

_(Henüz yok.)_

## Sabah raporu

_(Gece oturumu doldurur: biten görevler, kalanlar, taslak ablation tablosu, kullanıcının yapması gerekenler.)_
