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
| Tracing | Her LLM çağrısı ve her soru JSONL'e (soru, getirilen chunk'lar, prompt hash'i, token, gecikme, cache isabeti). JSONL esas kaynaktır; Langfuse yalnızca sonradan beslenen bir görüntüleme hedefi (T14) | Run'lar Langfuse'a bağımlı olmaz; kurumsal tarafta aranan observability yine gösterilir |
| CI | GitHub Actions: Windows + macOS matrisinde ruff ve pytest (GPU/DB/LLM testleri hariç) | Jenkins kişisel proje için ağır; aynı sinyal daha ucuz |
| Maliyet | Abonelik dolar göstermez; token × API liste fiyatı = "API-eşdeğeri maliyet" | Ablation'da maliyet sütunu kalsın |

### `claude_code` backend kuralları

- Her çağrı: tool'lar kapalı, tek tur (`max_turns=1`), özel system prompt (Claude Code preset'i değil), hiçbir ayar ya da CLAUDE.md yüklenmez, çalışma klasörü repo dışında geçici bir klasör. Seçenek adlarını güncel `claude-agent-sdk` dokümantasyonundan doğrula; tahmin etme.
- Alt süreç ortamından `ANTHROPIC_API_KEY` her zaman çıkarılır.
- Eşzamanlılık `LLM_MAX_CONCURRENCY` (varsayılan 4). Rate limit / kota hatasında üstel bekleme ve tekrar; 3 başarısızlıktan sonra o soru `error` olarak işaretlenir, run devam eder.
- JSON çıktı: prompt'ta şema + pydantic doğrulama + en fazla 1 tekrar.
- Disk cache (`.cache/llm/`, git dışı): anahtar = backend + model + system + user + şema hash'i. Aynı konfigürasyonun tekrar koşusu kota harcamaz.

## Gece oturumu çalışma protokolü

1. Bu plan kullanıcı tarafından onaylandı. (Görevler bu dokümanın son bölümüdür; `task-brief` son görevi dosya sonuna kadar alır.) Görevleri sırayla yap. Bir görev bitince kabul ölçütlerini komut çıktısıyla doğrula, görev başlığının altındaki `- [ ] Tamamlandı` kutusunu işaretle ve görevin commit mesajıyla commit at. **Push yok.**
2. Kullanıcıya soru sorma; uyuyor. Belirsizlikte makul bir varsayım yap ve tarih + gerekçeyle **Karar günlüğü**ne yaz.
3. Bloklanırsan (PDF inmiyor, Docker yok, kota bitti…) sorunu **Engeller**'e yaz ve ona bağımlı olmayan sıradaki göreve geç. Hiçbir şeyi sessizce atlama, hiçbir sonucu uydurma.
4. Gold set'te hiçbir soruyu `verified: true` yapma.
5. Ablation koşuları yalnızca **dev** split'inde. Test split'i yalnızca T12'de bir kez çalışır.
6. Kullanım limiti mesajı gelirse pencerenin sıfırlanmasını bekle ve devam et.
7. İş bitince ya da durmak zorunda kalınca **Sabah raporu**nu doldur.

### Orkestrasyon — alt agent'larla çalışma

Gece oturumu orkestratördür: kodu kendisi yazmaz, görevleri alt agent'lara verir ve sonuçları kendisi doğrular. Yöntem `subagent-driven-development` skill'i; aşağıdaki kurallar bu proje için skill'in önüne geçer.

- **Dal:** `main` üzerinde doğrudan çalışılır; worktree açılmaz (git dışı `data/`, `.env`, `.cache/` ve Docker volume'u worktree'de olmaz). Bu, skill'in "main'de başlamadan önce sor" adımının onayıdır.
- **Görev kimliği:** `task-brief PLAN.md T0` … `T13`. Global kısıtlar = `CLAUDE.md` "Kurallar" + bu dokümandaki "Mimari kararlar" ve "`claude_code` backend kuralları"; reviewer'a görevle ilgili olanlar aynen verilir.
- **Modeller:** tüm alt agent'lar (uygulayıcı, task reviewer, fix-loop, re-review) **`model: opus`** ile açılır; skill'in "en ucuz yeterli model" önerisi bu proje için geçersiz. Yalnızca en sondaki whole-branch review `fable` ile yapılabilir. Model her dispatch'te açıkça yazılır.
- **Commit:** görevin ana commit'i PLAN.md'deki commit mesajını aynen kullanır; fix-loop commit'leri Conventional Commits biçiminde, İngilizce. Commit'i uygulayıcı atar; `- [ ] Tamamlandı` kutusunu orkestratör, kabul ölçütlerini **kendisi** çalıştırıp gördükten sonra işaretler ve ayrı bir `docs: mark Tn done in plan` commit'iyle kaydeder.
- **Commit öncesi kontrol (orkestratör):** `.env`, `data/raw/`, `data/parsed/`, `runs/`, `.cache/` git'e girmiyor; `eval/gold/questions.jsonl`'da `verified: true` yok; PLAN.md'ye ya da README'ye yazılan her sayının bir `runs/` klasöründe karşılığı var.
- **Paralellik:** repoya aynı anda tek uygulayıcı yazar. Uzun veri işleri (Docling parse, embedding, gold set üretimi, eval koşuları) script'i hazır olduktan sonra orkestratör tarafından arka planda başlatılır; beklerken ona bağımlı olmayan sıradaki görev yürür (ör. T3 Docling çalışırken T5). Salt okuma araştırması (ör. T2'de PDF URL'leri) ayrı bir alt agent'ta paralel yapılabilir.
- **Karar ve engel kaydı:** skill'in ledger'ı (`.superpowers/sdd/`, git dışı) kurtarma haritasıdır. Kullanıcıyı ilgilendiren her `Ruling` ayrıca **Karar günlüğü**ne, her engel **Engeller**'e yazılır.
- **Skill'in durma koşulları:** geri alınamaz/yıkıcı işlem, güvenlik açısından hassas eylem ya da dışarıya etki (push, merge, yayın) gerekiyorsa sorma — kullanıcı uyuyor. O adımı yapma, Engeller'e yaz, sıradaki bağımsız göreve geç.
- **Bitiş:** skill'in final whole-branch review'u yapılır; `finishing-a-development-branch` adımı **uygulanmaz** (merge/push yok). Yerine **Sabah raporu** doldurulur.
- **Donanım:** GPU 4 GB VRAM. Modeller fp16 ve küçük batch'le, sırayla yüklenir. Docling GPU'da belleğe sığmaz ya da sayfa başına çok yavaşsa: önce yalnızca tablo içeren sayfalarda GPU, sonra CPU; seçim Karar günlüğüne.
- **Kota:** orkestratör, alt agent'lar ve pipeline'ın `claude_code` çağrıları aynı aboneliği kullanır. Gereksiz alt agent ve büyük çıktı okuma yok; büyük dosya/log okumayı alt agent'lar yapar, orkestratöre dosya yolu döner.

## Başlamadan önce — kullanıcının Windows'ta yapacakları

- [x] Repo Windows'a aktarıldı.
- [x] Docker Desktop kurulu ve açık: `docker info` hata vermiyor. _(2026-11-17: motor kapalıydı, başlatıldı; Server 27.5.1, 16 CPU, ~7.6 GB RAM)_
- [x] `nvidia-smi` GPU'yu gösteriyor. _(2026-11-17: RTX 3050 Laptop, 4 GB VRAM, sürücü 591.74, CUDA 13.1)_
- [x] `uv --version` çalışıyor (yoksa PowerShell'de `python -m pip install uv`). _(2026-11-17: yoktu, `pip install --user uv` ile kuruldu → uv 0.13.9; Python 3.11.2)_
- [x] Claude Code CLI kurulu ve aboneliğe giriş yapılmış: `claude -p "merhaba"` cevap veriyor. CLI yoksa PowerShell'de: `irm https://claude.ai/install.ps1 | iex`, ardından `claude` ile giriş. _(2026-11-17: CLI 2.1.5 → 2.1.318'e güncellendi; kullanıcı yeniden giriş yaptı; `claude -p --model claude-haiku-5-5` cevap veriyor, API anahtarı kullanılmıyor, abonelik `max`)_
- [x] `ANTHROPIC_API_KEY` tanımlı değil: PowerShell'de `echo $env:ANTHROPIC_API_KEY` boş dönüyor. _(2026-11-17: kullanıcı düzeyinde tanımlıydı, kullanıcı onayıyla kaldırıldı; User/Machine/Process üçü de boş)_
- [x] En az 30 GB boş disk. _(2026-11-17: C: 103 GB boş)_
- [ ] Uyku kapalı: Ayarlar → Sistem → Güç → prize takılıyken uyku "Hiçbir zaman". Bilgisayar prizde.
- [ ] Claude uygulaması yeniden başlatıldı (kaldırılan `ANTHROPIC_API_KEY` eski süreçlerde kalmasın).
- [ ] Yeni oturum **bypass permissions** modunda ve **Fable 5.2** ile açıldı (yoksa gece onay bekleyip durur).
- [x] `subagent-driven-development` skill'i kurulu (`~/.claude/skills/`). _(2026-11-17)_

### Gece oturumu başlatma promptu

```text
PLAN.md'yi, CLAUDE.md'yi ve eval/README.md'yi baştan sona oku. Bu planı onaylıyorum.
İlk iş bilgisayarın uyumasını engelle (keep-awake).

Sen orkestratörsün: kodu kendin yazma. Görevleri T0'dan başlayarak sırayla
subagent-driven-development skill'iyle alt agent'lara yaptır; PLAN.md'deki
"Orkestrasyon — alt agent'larla çalışma" kuralları skill'in önüne geçer (main'de çalış,
worktree açma; tüm alt agent'lar model: opus; finishing-a-development-branch yok).
Alt agent "bitti" dediğinde kabul ölçütlerini kendin çalıştırmadan görevi kapatma.

"Gece oturumu çalışma protokolü"ne uy: bana soru sorma, ben uyuyorum; varsayımlarını
Karar günlüğüne yaz, bloklanırsan Engeller'e yaz ve sıradaki bağımsız göreve geç.
Push yapma. Gold set'te hiçbir soruyu verified yapma. Bankaların resmi yatırımcı
ilişkileri sitelerinden ve KAP'tan bu plandaki 12 faaliyet raporu PDF'ini, ayrıca
planın gerektirdiği Python paketlerini, Docker imajını ve Hugging Face modellerini
indirmene izin veriyorum. İş bitince ya da durmak zorunda kalırsan PLAN.md'deki
Sabah raporunu doldur.
```

## Karar günlüğü

| Tarih | Karar | Gerekçe |
|---|---|---|
| 2026-11-17 | KAP faaliyet raporları; 4 banka × 2023–2025 | Hedef işverenlerin alanı; tablo, birim ve karşılaştırma zorlukları gerçek |
| 2026-11-17 | LLM'ler kullanıcının Claude aboneliği üzerinden (`claude_code` backend) | API bakiyesi gerekmesin; demo için `anthropic_api` backend'i de var |
| 2026-11-17 | Gece oturumu Windows + NVIDIA GPU makinesinde | Mac'te disk yetersiz (13 GB boş) |
| 2026-11-17 | README TR + EN; kod yorumları ve commit'ler İngilizce | Kullanıcı tercihi |
| 2026-11-17 | Gece oturumu Fable 5.2 orkestratör + `subagent-driven-development`; alt agent'lar varsayılan `opus`; `main`'de, worktree'siz | Kullanıcı tercihi; git dışı veri/.env/Docker volume worktree'de olmadığı için |
| 2026-11-17 | CI: GitHub Actions (T0). Langfuse: JSONL trace'lerin sonradan aktarımı (T14, P2). LiteLLM ve LangChain şimdilik yok | Kullanıcı tercihi; ablation için pipeline adımları elle yazılı kalır; LiteLLM `claude_code` backend'ini desteklemiyor |
| 2026-11-17 | T0 ortam kontrolü: `nvidia-smi` RTX 3050 Laptop 4 GB, sürücü 591.74, CUDA 13.1; `docker info` Server 27.5.1 (Docker Desktop); uv 0.13.9; Python 3.11.2; `claude -p "merhaba" --model claude-haiku-5-5` Türkçe cevap verdi, `ANTHROPIC_API_KEY` boş. torch: `ml` grubunda `pytorch-cu130` index'i yalnızca Windows/Linux'a marker'la bağlı (macOS PyPI'dan) → torch 2.14.1+cu130, `torch.cuda.is_available()` True. Build backend hatchling, `dev-mode-exact` + dev'de `editables` | cu130 sürücünün (CUDA 13.1) desteklediği en yeni PyTorch CUDA wheel'i; cu132 sürücü sürümünü aşıyor. Python 3.11 `.pth` dosyalarını yerel kodlamayla (cp1254) okuduğu için repo yolundaki `ş` düz editable kurulumu bozuyordu; exact mod yolu UTF-8 `.py` dosyasında tutuyor. uv'nin yönettiği Python 3.13 bu yolda kurulamadı (minor-version link hatası) |
| 2026-11-17 | Gece oturumu başında Docker motoru kapalıydı; orkestratör Docker Desktop'ı başlattı. Uyku için uygulamanın keep-awake'i istendi (oturum boşta kalana kadar boşta uykuyu engeller; kapak kapanırsa uyur). | Engeller'de öngörülmüştü; T0/T4 için şart |
| 2026-11-17 | Repo yolu ASCII dışı karakter (`ş`) içerdiği için Python 3.11 editable kurulumun `.pth` dosyasını okuyamıyor; build backend hatchling "exact" editable moduna alındı ve `editables` `dev` grubuna eklendi. | `import kaprag` başka türlü çalışmıyor; repo taşımak kullanıcının kararı. Bedeli: `uv sync --no-dev` ile paket import edilemez (bkz. Engeller) |
| 2026-11-17 | README'deki ablation tablosu: `eval/README.md` §6 "README'ye yalnızca doğrulanmış alt kümenin sayıları girer" diyor, T12 ise "Taslak — doğrulanmamış gold set" uyarılı tablo istiyor. T12 izlenecek: tablo tüm-set sayılarıyla, uyarı altında ve "doğrulanmış alt kümeyle değiştirilecek" notuyla yazılacak. | Bu gece hiçbir soru doğrulanamaz (agent `verified` yapmaz); T12 tam bu durum için yazılmış açık talimat. Yanlışsa bedeli: kullanıcı bir tabloyu siler |
| 2026-11-17 | T8 pipeline'ı metrikler için sıralı top-10 sonucu döndürür, üretim bağlamında top-5 kullanır. | T8 "top-5" diyor, `eval/README.md` §6 `recall@10`/`mrr@10` istiyor; ikisi birlikte ancak böyle ölçülür |
| 2026-11-17 | T6'daki "hibrit aramayla kontrol" (eval §7 adım 5) için dense ve FTS arama ilkelleri T4'te yazılır; hibrit bir eval konfigürasyonu olarak T9a'da ölçülür. Kabul satırı olmayan görevlerde (T8–T13) kabul = testler yeşil + görevin adını verdiği run/çıktı mevcut. | Plan sırası T6'yı T9a'dan önce koyuyor; kabul ölçütü boş bırakılmış görevler için asgari ölçüt |
| 2026-11-17 | API-eşdeğeri maliyette cache okuma/yazma token'ları düz girdi fiyatından sayılır. | `claude_code` backend'i kısa bir çağrıda ~2 girdi + ~640 cache-yazma token'ı raporluyor; yalnızca `input_tokens` sayılsa maliyet ~30 kat düşük çıkardı. Plan yalnızca girdi/çıktı fiyatı veriyor |
| 2026-11-17 | `LLMClient.complete` asenkron (`complete_sync` sarmalayıcısıyla); kalıcı hata `LLMCallError` fırlatır (önce trace'e yazılır), toplu koşucular yakalayıp soruyu `error` işaretler. `claude_code` backend'i `ANTHROPIC_API_KEY`'i süreç ortamından siler (SDK alt süreç ortamından değişken çıkaramıyor). | `claude-agent-sdk` asenkron; "run devam eder" kuralı çağıran tarafta uygulanır |
| 2026-11-17 | `page` her yerde (parse çıktısı, gold kanıt, atıf) 1 tabanlı **PDF sayfa indeksi**dir, basılı sayfa numarası değil. | Raporların çoğu bir PDF sayfasına iki basılı sayfa koyuyor; review arayüzü PDF sayfasını render ediyor |
| 2026-11-17 | Garanti korpusta yalnızca entegre faaliyet raporlarıyla kalır; ayrı BDDK finansal raporları eklenmez. Garanti `tablo` soruları raporda geçen rakamlarla sınırlıdır. | Garanti'nin entegre raporlarında tam bilanço/gelir tablosu yok; plan "bağımsız denetim raporları kapsam dışı" diyor. Yanlışsa bedeli: daha az Garanti tablo sorusu |
| 2026-11-17 | 12 raporun 12'si bankaların kendi yatırımcı ilişkileri sitelerinden indirildi (KAP gerekmedi); dört banka da 2023–2025 için yalnızca "Entegre Faaliyet Raporu" yayımlıyor. Toplam 3.187 PDF sayfası (`data/manifest.csv`). | T2; inmeyen doküman yok |
| 2026-11-17 | PyMuPDF çekirdek bağımlılıkta klasik sürüm hattına sabitlendi (`pymupdf>=1.26,<2`). | 2.x sürümü `onnxruntime`, `numpy` ve layout paketlerini çekirdeğe sürüklüyordu; T3 sayfa metni ve T7 sayfa görüntüsü için klasik sürüm yeterli, CI hafif kalıyor |
| 2026-11-17 | Docling çıktısında `level` ve `section_path` yaklaşıktır (başlık seviyeleri 20 sayfalık partiler içinde yazı stilinden çıkarılıyor); güvenilir olan en yakın başlıktır. T4'ün `structure` chunker'ı önek olarak en yakın başlığı kullanır. Grafik/infografik içindeki metin `in_picture` bayrağıyla korunur. | İlk Docling çıktısı grafik bölgelerindeki metni atıyordu: garanti-2023'te PyMuPDF'teki sayıların %90,5'i kalıyordu, düzeltmeden sonra %99,7 (`runs/2026-11-17-docling-picture-text/`). Seviyeleri yeniden hesaplamak spekülatif olurdu |
| 2026-11-17 | T3 parser gözlemi (her bankadan bir tablo sayfası: garanti-2023 s.69, akbank-2023 s.238, isbank-2023 s.146, yapikredi-2023 s.200; `runs/2026-11-17-parser-compare/compare.txt`): dört sayfada da PyMuPDF'teki sayıların hiçbiri Docling'de kayıp değil; PyMuPDF tabloyu hücre hücre alt alta düz metne döküyor (satır/sütun ilişkisi kayboluyor), Docling aynı sayfalarda 8–12 Markdown tablo ve başlıkları ayrı eleman olarak veriyor. Tüm korpusta Docling, PyMuPDF'teki sayısal token'ların %99,3–%100'ünü koruyor (`runs/2026-11-17-docling-full/coverage-vs-pymupdf.txt`); en zayıf doküman yapikredi-2024 (288 sayfanın 13'ünde sayıların %80'inden azı kalıyor). | T3 kabul ölçütü. Docling tam koşusu GPU'da 95,6 dk sürdü (3.015 sayfa, 1,90 s/sayfa, tepe 1,56 GB VRAM; `runs/2026-11-17-docling-full/parse.log`); CPU'ya düşmek gerekmedi. Log'da 21 tabloda yapı tahmini atılıp metin tek hücrede tutuldu, 83 uyarıda bazı hücreler tablodan düştü — tablo sorularında risk |
| 2026-11-17 | Docling entegrasyonunda iki geçici çözüm: (1) ASCII dışı yol yüzünden Docling'in PDF okuyucusu Windows 8.3 kısa yolu üzerinden yükleniyor; (2) satır sonunda bölünen sayı aralıklarının birleşmesini ("1999-"/"2002" → "19992002") önlemek için Docling'in iç fonksiyonu `PageAssembleModel.sanitize_text` sarmalanıyor. | (1) repo yolundaki `ş`; (2) aksi halde yıllar/sayılar bozuluyor. İkisi de Docling sürümü yükseltilince yeniden kontrol edilmeli (sürüm `uv.lock` ile sabit) |
| 2026-11-17 | `structure` chunker'ı: metin grupları tablo ya da grafik metniyle kesilince kapanır (bir chunk yalnızca kendi satırlarının sayfalarını kapsar); grafik içi metin sayfa başına ayrı `picture_text` chunk'ı olur; başlık satırı anlamsız (dejenere) tablolarda bölünen parçalarda başlık tekrarlanmaz. FTS sorgusu, sorudaki kelimelerin Türkçe kök/lexeme'lerinin OR'udur. | Eval'de isabet kuralı "gold sayfa chunk'ın sayfa aralığında" olduğu için şişmiş sayfa aralıkları `structure` lehine yanlı recall üretirdi; `plainto_tsquery` (AND) doğal dil sorularında boş dönüyor |
| 2026-11-17 | Eval harness (T5) kuralları: test split'i `--confirm-test` olmadan çalışmaz; `error` olan soru doğruluk metriklerinde yanlış sayılır ve ayrıca raporlanır; `answer_with_assumption` kontrolü deterministiktir (reddetmeden, gold esasla eşleşen dolu `basis`); gecikme yüzdeliklerinden cache isabetli sorular çıkarılır, maliyet cache isabetinde de ilk çağrının maliyetidir; judge maliyeti pipeline maliyetinden ayrı raporlanır. Şemada `expected.unit` ∈ {TL, %, adet}, `verified` ile `verified_at` birlikte dolar. | `eval/README.md` bu ayrıntıları açık bırakıyor; sayıların tekrar koşularda aynı çıkması ve test split'inin korunması için |
| 2026-11-17 | 30 pilot sorusu yalnızca dev split'inden seçilir. | Pilot sonucuna göre taslaklar ayarlanacağı için (eval §7 adım 9) pilot, test split'ine dokunmamalı; §2 yalnızca "tabakalı" diyor |
| 2026-11-17 | T7, T6'dan önce yapıldı (indeks kurulumu ve Docling sürerken). Review arayüzü kuralları: **Onayla** yalnızca o soruda `verified`/`verified_at` yazar; **Düzenle** tek başına doğrulamaz ("Kaydet" / "Kaydet ve onayla"), doğrulanmış bir soru düzenlenince eski `verified_at` kalır; **Reddet** satırı `questions.jsonl`'dan çıkarıp `eval/gold/rejected.jsonl`'a taşır; kullanıcının eklediği soru `author: human`, `verified: true` kaydedilir ve split'i (tip, banka) tabakasında 60:120 oranını koruyacak şekilde atanır (eşitlikte seed 42'li hash). Streamlit ayrı `app` grubunda (`uv sync --group ml --group app`). | T7'nin T6 çıktısına bağımlılığı yok; şema katı olduğu için "reddedildi" bayrağı yerine ayrı dosya; soruyu PDF'e bakarak yazan kullanıcı zaten doğrulamış olur; CI kurulumu hafif kalsın |
| 2026-11-17 | T4 indeksleri tam kuruldu (`runs/2026-11-17-index-build/`): `chunks_fixed` 10.488 satır, `chunks_structure` 25.864 satır (metin 16.097, tablo 9.073, grafik metni 694), 12'şer doküman; bge-m3 fp16, tepe 1,16 GB VRAM, kurulum 7,0 + 11,1 dk. "~500 token" bge-m3 tokenizer'ıyla ölçülür. Örnek sorgular (`runs/2026-11-17-index-sample-queries/queries.txt`): dense arama net kâr sorusunda Akbank 2024'ün özet yönetim kurulu raporu ve özet finansal sonuçlar sayfalarını (s.47, s.142), şube sorusunda şube sayısı geçen Garanti sayfalarını, sürdürülebilirlik sorusunda İş Bankası 2025'in ilgili sayfalarını getiriyor; FTS tek başına daha gürültülü (Türkçe kök bulucu "nın", "ne" gibi kısa lexeme'ler üretiyor, Postgres sıralamasında IDF yok). | T4 kabul ölçütü. Alt küme denemesinde tablo sorusunda cevabın bulunduğu tablo sayfası hiçbir kurulumda ilk 5'e girmemişti — baseline'ın ölçeceği zayıflık |
| 2026-11-17 | Gold set taslağı (T6): 243 soru, hepsi `author: llm`, `verified: false` (olgusal 39, tablo 60, hesaplama 26, karsilastirma 39, sentez 26, cevapsiz 33, belirsiz 20; dev 84 / test 159; 30 pilot, hepsi dev'de) — `runs/2026-11-17-gold-draft/`. Kurallar: esası (konsolide/solo) sayfadan anlaşılmayan TL/% tekil sorular atılır; türetilmiş soruların bütün aritmetiği Python'da; yıllar arası değişim en güncel raporun iki sütunlu sayfasından; `cevapsiz` sorularda `llm_checked` = korpus araması (dense + FTS, RRF) + LLM "cevaplanabilir mi" yargısı geçti demektir (kontrol edilecek sayfa yok); netleştirme isteyen `belirsiz` sorular `llm_checked: false`. | `eval/README.md` §4 konsolide varsayımı yüzünden esası bilinmeyen rakam sistematik yanlış puan üretirdi; §7 adım 3 cevapsız sorular için sayfa kontrolü tarif etmiyor |
| 2026-11-17 | Gold kanıt sayfaları tamamlandı (eval §3 "bilgi birden çok sayfada geçiyorsa hepsi yazılır"): aynı raporda rakam taraması + dense/FTS adayları, Haiku ile sayfa bazında kontrol; yalnızca kontrolü geçen sayfa eklenir (137 soruda 175 sayfa eklendi; cevaplanabilir soru başına ortalama 3,53 sayfa). **Sınırlama (pooling yanlılığı):** adayların bir kısmı ablation'da ölçülen aynı arama yöntemlerinden geliyor; hiçbir yöntemin getirmediği ve rakamı aynen basmayan sayfalar bilinmiyor. | İlk taslak soru başına tek sayfa yazıyordu; model doğru rakamı başka sayfadan bulunca retrieval/atıf ıskası sayılıyordu (baseline deneme koşusunda q062, q065) |
| 2026-11-17 | Tolerans sapması: yuvarlak basılmış rakamdan türeyen 4 gold soruda tolerans `eval/README.md` §3 kuralından geniş (q014, q023, q062: adet ±50.000; q092: rel 0,0193). | Kaynak rakam "25,4 milyon" / "2,6 trilyon" gibi yuvarlak; kesin rakamla verilen doğru cevap %1 / ±0 toleransla yanlış sayılırdı. Kullanıcı review'da kesin rakama çevirebilir |
| 2026-11-17 | Baseline (T8) kuralları: model kaynak **numarası** atıf yapar, pipeline bunu atıf yapılan chunk'ın sayfa aralığındaki `{doc, page}`'lere çevirir (yalnızca modelin atıf alanından; reddetme ve netleştirmede atıf yok); model rakamı basıldığı gibi metin olarak + ölçek kelimeli birimle verir, TL/%/adet normalizasyonu kodda yapılır; üretim hatası `error` olur, uydurma reddetme olmaz. | Model chunk içindeki sayfayı bilemez; "2.202" gibi binlik ayraçlı rakamın sayı olarak okunması 1000 kat hata üretirdi; hesaplar LLM'e bırakılmaz |
| 2026-11-17 | Pilot zorluk değerlendirmesi (eval §7 adım 9): baseline 30 pilot soruda %51,3 doğruluk (`runs/2026-11-17-baseline-pilot/`) → %15–%85 aralığında, set ne fazla kolay ne fazla zor; T6 taslakları ayarlanmadı. Baseline dev sonucu: 84 soruda %45,5 (`runs/2026-11-17-baseline-dev/`). | Plan T8 |
| 2026-11-17 | T9 kuralları: hibrit aramada her koldan ilk 40 aday, RRF k=60 (eşitlikte chunk kimliği sırası — alfabetik olarak önce gelen banka lehine, deterministik); reranker birleşik ilk 40 adayı sıralar, metrikler için ilk 10 ve üretim için ilk 5 yeniden sıralanmış listeden alınır; metadata filtresi `bank IN (…)` ve `year IN (soruda geçen yıllar ∩ korpus yılları)`, ayrıştırma başarısızsa filtresiz; filtreli vektör aramasında pgvector `hnsw.iterative_scan = strict_order`. T9 dört adım iki uygulayıcı dispatch'iyle yapıldı (a+b, c+d); her adımın kendi commit'i ve dev koşusu var. T9e (opsiyonel) yapılmadı. | Plan T9; dispatch sayısını azaltmak kota ve süre kazandırdı. Dense aramada 40'tan fazla aday istenince sessizce 40'ta kesen bir hata T9a'da bulunup düzeltildi (önceki koşuları etkilemiyor: en fazla 10 istiyorlardı) |
| 2026-11-17 | "T9'un en iyi pipeline'ı" = **`rerank`** (fixed chunk + hibrit + reranker): dev cevap doğruluğu 0,592 (`runs/2026-11-17-rerank-dev/`); structure 0,567, metadata_filter 0,575. T10 router'ı basit soruları `rerank`'e yollar; T12'de test split'inde baseline ile tek bir en iyi konfigürasyon çalıştırılır: `rerank`, ya da dev'de onu geçerse T10 router'ı. | Birincil ölçüt cevap doğruluğu. Üç konfigürasyon arasındaki fark 84 soruda gürültü içinde; daha basit ve recall'u en yüksek olan seçildi |
| 2026-11-17 | T10 router kuralları: sınıflandırma (Haiku) `simple` / `multi`, şüphede `simple`; `simple` yol `rerank` ile birebir aynı; `multi` yolda ayrıştırma (Sonnet) kapalı bir işlem kümesinden tek işlem ve en fazla 8 rakam araması üretir, yıllar arası değişimde iki rakam da son raporun karşılaştırmalı sütunundan okunur (eval §4); her arama `rerank` yığını + banka/rapor yılı SQL filtresiyle yapılır ve mevcut üretim/normalizasyon kodunu kullanır; bütün aritmetik Python'da (Decimal), birim ya da esas uyuşmazlığında ve eksik rakamda hesap yapılmaz, cevap eksik rakamı adıyla söyleyen bir reddetmedir; son cevabın `value`/`unit` alanı modelden değil hesaptan gelir. `langgraph` çekirdek bağımlılık (`langchain-core` ve `langsmith`'i de getiriyor). **Test split'inde "en iyi konfigürasyon" = `router`** (dev cevap doğruluğu 0,651; `rerank` 0,592). | Plan T10; "hesaplar LLM'e bırakılmaz". Dev sonucu `runs/2026-11-17-router-dev/` |
| 2026-11-17 | T11'de roller gerçek indeks tablolarına hemen yazılmadı; uygulayıcı kopya şemada doğruladı, gerçek tablolara atama test koşularından sonra yapılır. Görev sırası: T11, T10'dan önce; T14 (P2), T12'den önce başlatıldı. | `access_roles` güncellemesi ~15 bin satırın yeni sürümünü ve yeni HNSW girdilerini üretir; yaklaşık aramanın sonuçları ölçüm serisinin ortasında değişebilirdi. Sıra değişiklikleri boşta kalan uygulayıcı süresini kullanmak için |
| 2026-11-17 | T11 erişim modeli (simüle): chunk'ın `access_roles` alanı onu görebilen rolleri listeler — "Risk Yönetimi" başlığı altı → `{risk}`; yasal finansal raporun beşinci bölümü (dipnotlar) → `{analyst, risk}`; diğerleri → `{public}`. İsteğin rolü yalnızca çağırandan/konfigürasyondan gelir, soru metninden asla; etkin rol kümesi her zaman `public`'i içerir. Sabit chunk'larda roller sayfa düzeyinde ve kapsanan sayfaların en kısıtlısıdır. Mevcut eval konfigürasyonları erişim filtresiz çalışır. | Plan T11. "Dipnot" başlık kuralı bazı bankalarda finansal tabloların tamamını ve denetçi raporunu da yakaladığı için dipnot tanımı beşinci bölümle sınırlandı. Kısıtlı pay büyük (Garanti hariç raporların yaklaşık %40–55'i): `public` rolüyle demo sık sık reddeder. Hatalar yalnızca aşırı kısıtlama yönünde |
| 2026-11-17 | T13 kuralları: API (`scripts/serve_api.py`) ve sohbet demosu (`scripts/chat_app.py`) ortak bir servis katmanını kullanır, varsayılan konfigürasyon `router`; rol yalnızca API parametresinden / arayüz seçicisinden gelir; ikisi de varsayılan olarak 127.0.0.1'e bağlanır, kimlik doğrulama yok; abonelik backend'i (`claude_code`) ile localhost dışına açılmayı API de demo da reddeder. `.streamlit/config.toml` repodaki bütün Streamlit uygulamalarını (review arayüzü dahil) localhost'a bağlar. İngilizce sorular ana gold dosyasına eklenmedi: `eval/gold/questions_en.jsonl` (15 soru, dev sorularının çevirisi, `verified: false`). | "Başkalarına açılan demo abonelikle değil API anahtarıyla çalışır" kararı; bu gecenin gold seti bayt bayt aynı kalsın. İngilizce deneme (`runs/2026-11-17-router-english-probe/`): cevap puanı 0,720, aynı soruların Türkçe asıllarında 0,653; page_recall@5 0,769'a karşı 0,846; n=15, fark gürültü. 15 İngilizce sorunun 6'sı Türkçe cevaplandı (prompt'lara dokunulmadı) |
| 2026-11-17 | Oturum boyunca uykuyu engellemek için uygulamanın keep-awake'ine ek olarak arka planda bir PowerShell süreci çalıştırıldı (`SetThreadExecutionState`; hiçbir güç ayarı değişmedi; oturum sonunda durduruldu). Alt agent'ların commit'leri `Co-Authored-By: Claude Opus 5.5`, orkestratörün `docs:` commit'leri `Claude Fable 5.2` satırı taşıyor. | Uygulamanın keep-awake'i oturum ~5 dk boşta kalınca bırakıyor, veri işleri ise gözetimsiz saatlerce sürdü; commit satırı işi yapan modeli doğru göstersin |
| 2026-11-18 | Final whole-branch review `fable` ile yapıldı; bulguları tek bir düzeltme dalgası ve tek bir kapsamlı re-review ile kapatıldı. Dev/test kaynak örtüşmesi (159 test sorusunun 61'i) düzeltilmedi, README'lere sınırlama olarak yazıldı; split'i yeniden yapmak kullanıcının kararı. Task review'ların ertelediği küçük bulgulardan yalnızca ölçümleri değiştirmeyenler düzeltildi. | Skill kuralı: tek düzeltme dalgası. Split'i değiştirmek bu gecenin bütün dev/test sayılarını geçersiz kılardı ve gold dosyası review'dan önce bayt bayt sabit kalmalı |
| 2026-11-18 | **Kullanıcı kararı:** gold set şimdilik elle doğrulanmayacak; bağımsız LLM agent'ları denetledi, sonuç ayrı bir `audit` alanına (`passed` / `fixed` / `flagged`) yazıldı, `verified` insan review'una ayrılmış olarak kaldı. README'ler "LLM ile üretildi ve bağımsız bir LLM denetiminden geçti; insan denetimi yapılmadı" diyor; denetimin insan doğrulaması gibi gösterilmesi ya da gizlenmesi yapılmadı. `eval/README.md` §3 ve §6'ya alan ve "denetlenmiş alt küme" eklendi. | Kullanıcı 4–8 saatlik review'u yapmak istemedi; işverene yanlış bilgi vermemek için etiket dürüst tutuldu |
| 2026-11-18 | Denetim kararlarının uygulanma kuralı (`runs/2026-11-18-gold-audit/decisions.json`): sorunsuz → `passed` (cevabı içermeyen ek kanıt sayfaları çıkarılır, ilk kanıt sayfası korunur); soru metni değişmeden düzeltilebilen → denetçinin önerdiği alanlar uygulanır, `fixed`; soru metninin yeniden yazılması gereken → dokunulmaz, `flagged`; tekrar ya da çelişkili → `rejected.jsonl`. Koşular yeniden çalıştırılmaz, kayıtlı çıktılar yeniden puanlanır. | Kayıtlı sistem cevapları eski soru metnine ait olduğu için soru metni değiştirilemez; test split'i bir kez koşulur ve indeks (roller yazıldıktan sonra) değişti |

## Engeller

| Tarih | Engel | Durum / yapılacak |
|---|---|---|
| 2026-11-17 | `ANTHROPIC_API_KEY` Windows kullanıcı ortam değişkeni olarak tanımlı (User scope). Yeni açılan her terminal ve Claude Code oturumu bunu devralır; `claude_code` backend'inde abonelik yerine API faturalanabilir. | **Çözüldü (2026-11-17):** kullanıcı onayıyla User scope'tan kaldırıldı. Açık terminaller/Claude uygulaması eski değeri hâlâ taşıyabilir; gece oturumundan önce yeniden başlatılmalı. Anahtar `anthropic_api` backend'i için gerekiyorsa yalnızca `.env`'de tutulmalı. |
| 2026-11-17 | GPU yalnızca 4 GB VRAM (RTX 3050 Laptop). bge-m3 + bge-reranker-v2-m3 + Docling aynı anda belleğe sığmayabilir. | Engel değil, risk: modelleri fp16 ve küçük batch'le, sırayla yükle; Docling'i embedding'den ayrı çalıştır. |
| 2026-11-17 | Docker Desktop'a ~7.6 GB RAM ayrılmış; Docker Desktop oturum açılışında otomatik başlamıyor olabilir (bugün kapalıydı). | Gece oturumundan önce Docker Desktop'ın açık olduğunu kontrol et. |
| 2026-11-17 | `claude -p "merhaba"` başarısız: "OAuth session expired and could not be refreshed". CLI güncellendi (2.1.318), sorun sürüyor. `claude-agent-sdk` aynı girişi kullandığı için T1 ve tüm LLM adımları buna bağlı. | **Çözüldü (2026-11-17):** kullanıcı yeniden giriş yaptı; Haiku ile `claude -p` doğrulandı. |
| 2026-11-17 | Uyku ayarı kontrol edilmedi. | Kullanıcı kontrol etmeli. |
| 2026-11-17 | Repo yolundaki `ş` yüzünden paket yalnızca `dev` grubundaki `editables` ile import edilebiliyor; `uv sync --no-dev` (ör. ileride bir prod/Docker kurulumu) paketi import edilemez bırakır. uv'nin yönetilen Python 3.13'ü bu kullanıcı yolunda kurulamadı. | Engel değil, risk. Kalıcı çözüm kullanıcının kararı: repoyu ASCII bir yola taşımak ya da Python 3.13+. CI ve yerel geliştirme etkilenmiyor. |
| 2026-11-17 | T14: Langfuse self-host yığını ayağa kaldırılmadı; exporter gerçek bir Langfuse sunucusuna karşı hiç çalıştırılmadı (yalnızca birim testleri, sahte istemci/exporter ve `--dry-run`). | Orkestratör bilerek yapmadı: yığın `.env`'e üretilmiş sırlar yazılmasını, bir yönetici kullanıcı ve API anahtarı sağlanmasını ve birkaç GB imaj indirilmesini gerektiriyor (Docker'a ~7,6 GB RAM ayrılmış). **Kullanıcı yapacak:** `.env`'deki boş `LANGFUSE_*` değerlerini doldur (`.env.example`'daki açıklamalara göre) → `docker compose --profile observability up -d` (arayüz http://localhost:3300; anahtarlar ilk açılışta `.env`'den oluşur, arayüz girişi gerekmez) → `uv sync --group ml --group app --group observability` → `uv run python scripts/export_langfuse.py --run runs/2026-11-17-rerank-dev` → trace sayısının 84 olduğunu kontrol et. Aynı run 15–30 s içinde iki kez aktarılırsa çift kayıt oluşabilir. |
| 2026-11-17 | Abonelik penceresi (5 saatlik / 7 günlük) dolarsa `claude_code` çağrıları 2 s + 4 s beklemeyle 3 denemeden sonra `error` olur; pencerenin sıfırlanmasını bekleme istemcide yok. | Engel değil, risk. Başarısız çağrılar cache'lenmediği için run yeniden çalıştırılınca yalnızca hatalı sorular kota harcar; orkestratör run'lardaki `error` sayısına bakıp gerekirse pencere sıfırlanınca tekrar koşar. |

## Sabah raporu

_Gece oturumu 2026-11-17 15:50'de başladı, 2026-11-18 00:20'de bitti. Hiçbir şey push edilmedi; gold set'te `verified: true` yok._

### Güncelleme — 2026-11-18, agent denetimi (kullanıcı kararıyla)

Kullanıcı gold set'i şimdilik elle doğrulamamaya karar verdi. Bunun yerine bağımsız LLM agent'ları 243 taslağın hepsini render edilmiş PDF sayfa görüntülerine karşı denetledi (`runs/2026-11-18-gold-audit/`). Sonuç `verified` alanına değil ayrı bir `audit` alanına yazıldı; **hiçbir soru insan tarafından doğrulanmadı**.

- **Denetim sonucu (243 taslak):** 224 sorunsuz; 13 düzeltildi (11'inde yalnızca kanıt sayfaları, 3'ünde beklenen cevap metni, birer tanesinde değer ve tolerans); 3'ü insan kararı gerektiriyor ve olduğu gibi bırakıldı (q080, q090, q239 — soru metni belirsiz; denetlenmiş alt kümede değiller); 3'ü reddedildi ve `eval/gold/rejected.jsonl`'a taşındı (q063 ve q078 tekrar, q068'de rapor iki farklı çalışan sayısı veriyor). Ayrıca cevabı içermeyen 25 ek kanıt sayfası çıkarıldı, türetilmiş 9 sorunun kanıtı kaynaklarından yeniden hesaplandı.
- **Denetimin bulgusu:** kontrol edilen finansal rakamlarda değer, ölçek, esas ya da yıl sütunu hatası çıkmadı; asıl kusur otomatik eklenen ve cevabı içermeyen kanıt sayfalarıydı (en ağırı q091: gösterilen beş sayfanın hiçbirinde rakam yoktu).
- **Gold set şimdi:** 240 soru (dev 84, test 156); denetlenmiş alt küme 237 soru (dev 83, test 154).
- **Koşular yeniden çalıştırılmadı.** Kayıtlı sistem çıktıları denetlenmiş gold set'e karşı yeniden puanlandı (`scripts/rescore_run.py`, klasörler `runs/<özgün>-rescored/`; toplam 3 yeni judge çağrısı). Test split'i hâlâ bir kez koşulmuş durumda.

**Denetlenmiş alt küme — test (154 soru):**

| Konfigürasyon | Cevap doğruluğu | Sayısal doğruluk | page_recall@5 | page_recall@10 | Yanlış reddetme | Maliyet/soru | p50 / p95 gecikme | Run |
|---|---|---|---|---|---|---|---|---|
| baseline | 0,487 | 0,389 | 0,605 | 0,698 | 0,445 | $0,0014 | 6,3 s / 12,3 s | `runs/2026-11-17-baseline-test-rescored/` |
| router (en iyi) | 0,681 | 0,537 | 0,829 | 0,899 | 0,134 | $0,0048 | 12,0 s / 25,8 s | `runs/2026-11-17-router-test-rescored/` |

**Denetlenmiş alt küme — dev (83 soru):**

| Konfigürasyon | Cevap doğruluğu | page_recall@5 | page_recall@10 | Kaynak doğruluğu (precision) | Yanlış reddetme | Run |
|---|---|---|---|---|---|---|
| baseline | 0,460 | 0,575 | 0,753 | 0,35 | 0,500 | `runs/2026-11-17-baseline-dev-rescored/` |
| + hybrid | 0,505 | 0,671 | 0,781 | 0,47 | 0,455 | `runs/2026-11-17-hybrid-dev-rescored/` |
| + rerank | 0,599 | 0,822 | 0,918 | 0,45 | 0,364 | `runs/2026-11-17-rerank-dev-rescored/` |
| + structure | 0,574 | 0,726 | 0,767 | 0,70 | 0,333 | `runs/2026-11-17-structure-dev-rescored/` |
| + metadata_filter | 0,582 | 0,767 | 0,822 | 0,63 | 0,333 | `runs/2026-11-17-metadata-filter-dev-rescored/` |
| router | 0,659 | 0,918 | 0,945 | 0,43 | 0,197 | `runs/2026-11-17-router-dev-rescored/` |

Denetim cevap doğruluğunu konfigürasyon başına en fazla 0,01–0,02 oynattı; sıralama ve yorumlar değişmedi. Router test koşusundaki judge hatası (q127) yeniden puanlamada giderildi. Dev/test örtüşmesi denetlenmiş sette: 156 test sorusunun 61'i, 26 test `karsilastirma` sorusunun 24'ü; router örtüşenlerde 0,59, diğerlerinde 0,73 (`runs/2026-11-17-split-overlap/overlap-after-audit.txt`).

Aşağıdaki gece raporu tabloları denetimden **önceki** taslak set (243 soru) üzerindedir; tarihçe olarak duruyor.

Hâlâ senin kararını bekleyenler: işaretli üç sorunun yeniden yazılması (q080, q090, q239); istersen 40–50 soruluk bir insan denetimi (README'deki ifade "N soruluk insan denetiminde uyum %X" olarak güçlenir); split kararı; judge kalibrasyonu; push.

**Biten görevler:** planın 15 görevinin hepsi (T0–T14). Her görevin kabul ölçütlerini orkestratör kendisi çalıştırdı; her görev ayrı bir task review'dan, bütün dal en sonda bir whole-branch review'dan geçti. Whole-branch review kritik bulgu çıkarmadı; dört önemli bulgu ve birkaç küçük düzeltme tek bir düzeltme dalgasıyla giderildi (aşağıda "Final review").

**Yapılmayanlar (bilerek):**
- T9e (opsiyonel embedding değişimi) çalıştırılmadı.
- T14'ün opsiyonel adımı: Langfuse yığını ayağa kaldırılmadı, exporter gerçek bir sunucuya karşı denenmedi (Engeller'e bakınız).
- Judge kalibrasyonu (`eval/README.md` §6): 50 elle etiketli cevap gerektiriyor.
- CI GitHub'da koşmadı (push yok); macOS'ta hiçbir şey denenmedi.

**Taslak sonuçlar — doğrulanmamış gold set.** Hiçbir soru `verified` değil. `eval/README.md` §6'ya göre README/CV'ye yalnızca doğrulanmış alt kümenin sayıları girer; README'lerdeki tablolar T12'nin istediği gibi "Taslak — doğrulanmamış gold set" etiketiyle duruyor ve doğrulamadan sonra değiştirilmeli.

| Konfigürasyon | Split | Soru | Cevap doğruluğu | page_recall@5 | page_recall@10 | Yanlış reddetme | Maliyet/soru (API-eşdeğeri) | p50 gecikme | Run |
|---|---|---|---|---|---|---|---|---|---|
| baseline (fixed, dense, top-5, Haiku) | pilot | 30 | 0,513 | 0,679 | 0,857 | 0,458 | $0,0013 | 6,8 s | `runs/2026-11-17-baseline-pilot/` |
| baseline (fixed, dense, top-5, Haiku) | dev | 84 | 0,455 | 0,581 | 0,770 | 0,500 | $0,0013 | 7,8 s | `runs/2026-11-17-baseline-dev/` |
| + hybrid (dense + Türkçe FTS, RRF k=60) | dev | 84 | 0,499 | 0,689 | 0,770 | 0,455 | $0,0014 | 6,7 s | `runs/2026-11-17-hybrid-dev/` |
| + rerank (ilk 40 → bge-reranker-v2-m3) | dev | 84 | 0,592 | 0,811 | 0,919 | 0,364 | $0,0014 | 9,5 s | `runs/2026-11-17-rerank-dev/` |
| + structure (Docling tabanlı indeks) | dev | 84 | 0,567 | 0,730 | 0,770 | 0,333 | $0,0012 | 7,9 s | `runs/2026-11-17-structure-dev/` |
| + metadata_filter (Haiku banka/yıl → SQL filtresi) | dev | 84 | 0,575 | 0,770 | 0,838 | 0,333 | $0,0013 | 10,3 s | `runs/2026-11-17-metadata-filter-dev/` |
| router (LangGraph: basit → `rerank`; hesaplama/karşılaştırma → ayrıştır + Python hesap) | dev | 84 | 0,651 | 0,905 | 0,932 | 0,197 | $0,0045 | (bkz. not) | `runs/2026-11-17-router-dev/` |

**Test split'i (tek sefer çalıştırıldı; 159 soru, hata yok):**

| Konfigürasyon | Cevap doğruluğu | Sayısal doğruluk | page_recall@5 | page_recall@10 | Yanlış reddetme | Doğru reddetme | Maliyet/soru | p50 / p95 gecikme | Run |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 0,478 | 0,373 | 0,604 | 0,709 | 0,435 | 1,00 | $0,0014 | 6,4 s / 12,7 s | `runs/2026-11-17-baseline-test/` |
| router (en iyi) | 0,664 | 0,508 | 0,813 | 0,896 | 0,137 | 1,00 | $0,0047 | 12,0 s / 25,7 s | `runs/2026-11-17-router-test/` |

Test'te tip bazında cevap doğruluğu — baseline: olgusal 0,69 · tablo 0,37 · hesaplama 0,29 · karsilastirma 0,00 · sentez 0,35 · cevapsiz 1,00 · belirsiz 0,86; router: olgusal 0,81 · tablo 0,42 · hesaplama 0,65 · karsilastirma 0,56 · sentez 0,53 · cevapsiz 1,00 · belirsiz 0,93.

Test notları: router koşusunda bir sorunun (q127) judge çağrısı geçersiz JSON döndürdü; o soru cevap doğruluğuna girmedi (n=158). Router gecikmesi 130 soruda ölçüldü (29 soruda bir alt çağrı cache'ten geldi). Her iki koşu aynı pipeline koduyla (`3de913f`) çalıştı; kayıtlardaki `-dirty` işareti o sırada commit edilmemiş T14 dosyalarından geliyor (`runs/logs/2026-11-17-test-runs.provenance.txt`).

**Erişim kontrolü (T11, simüle):** roller gerçek tablolara test koşularından sonra yazıldı (`runs/2026-11-17-acl-assign/`): `chunks_fixed` %61,8 public / %10,3 analyst+risk / %27,9 risk; `chunks_structure` %56,5 / %16,7 / %26,8. Sızıntı testi (`eval/acl_leak.jsonl`, 30 saldırgan sorgu, `public` rolü, getirilen ilk 40 sonuç kontrol edilir): `rerank` 0/30 sızıntı, `structure` 0/30 sızıntı; pozitif kontrol (filtre kapalı) 30/30 sorguda kısıtlı chunk getiriyor (`runs/2026-11-17-acl-leak-rerank-public/`, `runs/2026-11-17-acl-leak-structure-public/`).

Dev'de tip bazında cevap doğruluğu:

| Konfigürasyon | olgusal | tablo | hesaplama | karsilastirma | sentez | cevapsiz | belirsiz |
|---|---|---|---|---|---|---|---|
| baseline | 0,69 | 0,41 | 0,11 | 0,08 | 0,36 | 1,00 | 0,50 |
| + hybrid | 0,77 | 0,45 | 0,00 | 0,08 | 0,43 | 1,00 | 0,83 |
| + rerank | 0,77 | 0,55 | 0,11 | 0,31 | 0,63 | 1,00 | 0,83 |
| + structure | 0,85 | 0,41 | 0,11 | 0,38 | 0,52 | 1,00 | 0,83 |
| + metadata_filter | 0,85 | 0,45 | 0,11 | 0,38 | 0,48 | 1,00 | 0,83 |
| router | 0,77 | 0,55 | 0,33 | 0,54 | 0,63 | 1,00 | 0,83 |

Atıf metrikleri (dev): atıf kesinliği baseline 0,37 · hybrid 0,47 · rerank 0,46 · structure 0,74 · metadata_filter 0,67; atıf recall'u 0,19 · 0,25 · 0,36 · 0,36 · 0,39.

Okuma notları:
- Dev seti 84 soru; tek bir oranın standart hatası yaklaşık 5 puan. rerank / structure / metadata_filter arasındaki 1–3 puanlık farklar gürültü içinde; baseline → rerank farkı (yaklaşık 14 puan) değil.
- `structure` cevap doğruluğunu artırmadı, atıf kesinliğini belirgin artırdı (chunk'lar tek sayfalık). Sayfa recall'undaki düşüşün bir kısmı ölçümden kaynaklanıyor olabilir: `fixed` chunk'lar birkaç sayfaya yayıldığı için "gold sayfa chunk'ın sayfa aralığında" kuralını daha kolay tutturur. Bu doğrulanmadı.
- Hiçbir retrieval adımı `hesaplama` sorularını çözmedi (0,00–0,11); `karsilastirma` 0,08'den 0,38'e çıktı. Bunlar T10'daki router'ın hedefi.
- Gecikme cache isabetli sorular hariç ölçülür (baseline dev'de 84 sorunun 54'ü, metadata_filter'da 74'ü, diğerlerinde 79'u). Reranker arama süresine soru başına yaklaşık 1,4 s, metadata filtresi bir Haiku çağrısı ekliyor.
- T9e (embedding'i `multilingual-e5-large` ile değiştirmek, opsiyonel) yapılmadı.
- Router (T10): dev'de 9 hesaplama ve 13 karşılaştırma sorusunun hepsini agent yoluna, diğer 62 soruyu basit yola (`rerank` ile birebir aynı cevaplar) yolladı. Maliyet soru başına yaklaşık 3 kat (agent yolunda bir Sonnet çağrısı + alt soru başına Haiku). Router'ın dev gecikmesi diğer satırlarla karşılaştırılamaz: basit yoldaki cevaplar cache'ten geldiği için gecikme yalnızca 9 soruda ölçüldü (p50 25,0 s, ağırlıkla agent yolu). Agent yolundaki hataların kaynağı tek rakam çıkarma/normalizasyon adımı (ör. "3 trilyon 2 milyar" yazımı normalize edilemiyor, yanlış tanımlı oran seçiliyor).

**Dev ve test bağımsız değil (final review bulgusu):** ayrım (tip, banka) tabakalı ve `derived_from`'u hesaba katmıyor. 159 test sorusunun 61'i bir dev sorusuyla aynı kaynak rakamı paylaşıyor; 26 test `karsilastirma` sorusunun 24'ü böyle (`runs/2026-11-17-split-overlap/overlap.txt`). Router dev'de seçildi ve hedefi tam bu tipler. Zarar kanıtı yok: gold içeriği hiçbir prompt'ta ya da pipeline kodunda geçmiyor, router test'te örtüşen sorularda 0,59, diğerlerinde 0,71 alıyor. Yine de test seti bu tipler için tam "görülmemiş" sayılamaz; README'lere sınırlama olarak yazıldı.

**Final review (whole-branch, Fable):** doğruladıkları — erişim rolü router'ın alt aramaları ve geri düşüş yolları dahil her aramaya uygulanıyor ve hiçbir yerde soru metninden gelmiyor; bütün SQL parametreli; `verified` yalnızca review arayüzünün üç açık eyleminde yazılıyor; kontrol ettiği bütün sayılar koşu dosyalarıyla eşleşiyor; test split'i konfigürasyon başına bir kez koşulmuş; `3de913f`'ten sonra pipeline, konfigürasyon ve gold dosyalarında değişiklik yok. Düzeltme dalgasında giderilenler: `draft_gold.py` artık var olan gold dosyasının üzerine yazmayı reddediyor (doğrulanmış ya da insan yazımı soru varsa `--overwrite` ile bile); veritabanı portu compose dosyasında 127.0.0.1'e bağlandı; router yolunda çağıranın rolünü sabitleyen testler eklendi; erişim kuralındaki "bölümde/bölümün" kenar durumu kapatıldı (atanmış rollerde değişen satır yok); sızıntı testi boş sonuç listesinde artık başarısız oluyor; `--split pilot` test sorusu içeriyorsa `--confirm-test` istiyor; cache'teki bozuk bir cevap artık cache ıskası sayılıyor; README'lere dev/test örtüşmesi ve iki ölçüm notu eklendi. Hiçbiri ölçülmüş sonuçları değiştirmiyor; dalga ayrı bir re-review'dan geçti (dokuz maddenin dokuzu giderildi, yeni önemli bulgu yok) ve sonrasında 956 test yeşil. Kalan küçük not: gold üzerine yazma koruması yalnızca `scripts/draft_gold.py`'de, pipeline doğrudan Python'dan çağrılırsa korumuyor.

**Bilinen sorunlar (düzeltilmedi; ölçümleri ya da gold dosyasını değiştirirdi ya da küçük):**
- Gold taslağında: q015 ve q063 aynı şube sayısı sorusuna 804 ve 805 diyor; q078 ve q079 aynı soru; q092 yuvarlak "2,6 trilyon" kullanıyor (kesin rakam s.204'te, kanıtta yok); q031'de `basis` İş Bankası'na değil TSKB'ye ait; q167 ve q174'ün kanıt sayfalarından biri konu dışı; sayfadan üretilen bazı `tablo` soruları fazla ayrıntılı dipnot kalemleri soruyor; `cevapsiz` ve metrik soruları birkaç kalıbı tekrarlıyor.
- Router'ın agent yolundaki hataların kaynağı tek rakam çıkarma/normalizasyon: "3 trilyon 2 milyar" gibi yazımlar normalize edilemiyor, aynı metriğin farklı tanımlı değeri seçilebiliyor. Router'ın reddetme ve şablon cevapları İngilizce soruda da Türkçe.
- 15 İngilizce sorunun 6'sı Türkçe cevaplandı (prompt kuralına rağmen).
- Erişim kuralı sayfa düzeyinde ve yalnızca aşırı kısıtlama yönünde hatalı (ör. yapikredi-2024 yaklaşık s.208–217); Docling satırı olmayan sayfalar `public` sayılıyor (korpusta böyle metinli sayfa yok).
- `scripts/serve_api.py` açılma korumasını `kaprag.serve.exposure` yerine kendi içinde tekrar ediyor (ikisi küçük ayrıntılarda farklı); `scripts/chat_app.py`'de `$KAPRAG_SERVE_FACTORY` ile keyfi modül yükleyen bir test kancası var.
- Test koşularının köken dosyasında (`runs/logs/2026-11-17-test-runs.provenance.txt`) "başlatma anındaki git status" bölümü boş; `-dirty` açıklaması sonradan yazılan nota ve git geçmişine dayanıyor.
- Elle yeniden adlandırılan deneme koşu klasörleri (`*-smoke*`, `*-cachecheck*`, `router-english-probe`) içlerinde eski `run_id`'yi taşıyor.
- Task review'ların ertelediği küçük bulguların tam listesi ve triyajı: `.superpowers/sdd/PLAN/final-review-deferred-minors.md` (git dışı çalışma klasörü; bu oturumun defteri `progress.md` de orada).

**Kullanıcının yapması gerekenler (öncelik sırasıyla):**
1. Gold set'i review arayüzünden doğrula: `uv sync --group ml --group app --group observability`, sonra repo kökünden `uv run streamlit run scripts/review_app.py`. 243 taslak var; hedef 180 soru ve en az %35'i insan yazımı. Her review oturumundan sonra `eval/gold/questions.jsonl`'ı commit'le.
2. Doğrulanmış sayıları yayımlamadan önce split kararını ver: `derived_from` gruplarına göre yeniden ayırmak test'i gerçekten görülmemiş yapar ama bu gecenin dev ve test sayılarını geçersiz kılar; olduğu gibi bırakmak README'deki sınırlamayı taşımak demek.
3. `docker compose up -d` çalıştır: veritabanı container'ı yeniden oluşup 127.0.0.1'e bağlanır (şu an hâlâ 0.0.0.0:5432'de dinliyor; volume indeksi korur).
4. Judge kalibrasyonu için 50 cevabı elle etiketle (`eval/README.md` §6).
5. İlk `git push`'ta GitHub Actions'ın Windows ve macOS'ta yeşil olduğunu kontrol et (hafif kurulum yerelde yalnızca Windows'ta denendi). `main`, `origin/main`'in önünde; push senin kararın.
6. Langfuse'u denemek istersen Engeller'deki adımları izle.
7. Repo yolundaki `ş` için karar ver (Engeller): repoyu ASCII bir yola taşımak iki geçici çözümü gereksiz kılar.
8. `eval/README.md` §6'ya başlık metriğinin tanımını eklemeyi düşün: "cevap doğruluğu" bileşik bir puan (tanımı şu an yalnızca `src/kaprag/eval/metrics.py` docstring'inde ve README'lerde).
9. q127'nin judge hatası router test sonucunu n=158'de bırakıyor; bunu düzeltmek için test split'ini yeniden koşma.

Uygulamaları başlatma: review arayüzü `uv run streamlit run scripts/review_app.py`; sohbet demosu `uv run streamlit run scripts/chat_app.py`; API `uv run python scripts/serve_api.py` (üçü de repo kökünden, 127.0.0.1'de). Ayrıntılar `README.md`'de.

## Görevler

Öncelik: **P0** sabaha kadar şart, **P1** olması beklenen, **P2** zaman kalırsa.

### Task T0 — Ortam ve iskelet (P0)
- [x] Tamamlandı
- Donanım ve araçları kontrol et (`nvidia-smi`, `docker info`, `uv`, `python`, `claude -p`); sonucu Karar günlüğüne yaz.
- `pyproject.toml` (uv), `src/kaprag/`, `tests/`, `scripts/`, ruff ve pytest ayarları; CUDA'lı torch (CUDA sürümünü `nvidia-smi`'den oku).
- `docker-compose.yml`: `pgvector/pgvector:pg17`, port 5432, isimli volume. `.env.example`'dan `.env` oluştur.
- Ağır bağımlılıklar (torch, sentence-transformers, docling) ayrı bir uv dependency group'unda (`ml`); hafif çekirdek + dev araçları onsuz kurulabilir. GPU, veritabanı ya da Claude gerektiren testler pytest marker'larıyla (`gpu`, `db`, `llm`) işaretlenir.
- GitHub Actions `.github/workflows/ci.yml`: push ve pull request'te, `windows-latest` ve `macos-latest` matrisinde `astral-sh/setup-uv` → `uv sync --locked` (`ml` grubu hariç) → `uv run ruff check .` → `uv run pytest -m "not gpu and not db and not llm"`. Action'ların güncel ana sürümlerini dokümantasyondan doğrula; tahmin etme. Sır kullanılmaz.
- **Kabul:** `docker compose up -d` sonrası `CREATE EXTENSION IF NOT EXISTS vector` çalışıyor; `uv run pytest` yeşil; `torch.cuda.is_available()` → `True`; CI'daki komut dizisi yerelde `ml` grubu olmadan temiz bir ortamda (ör. geçici `UV_PROJECT_ENVIRONMENT`) aynen geçiyor; workflow YAML'ı geçerli. (Push yok; CI'ın GitHub'da gerçekten koşması kullanıcının ilk push'unda görülür.)
- **Commit:** `chore: scaffold project with uv, pgvector compose, ruff, pytest and CI`

### Task T1 — LLM istemcisi (P0)
- [x] Tamamlandı
- `kaprag.llm`: `LLMClient.complete(system, user, model, schema=None) -> LLMResult(text, parsed, usage, latency_ms, cached)`.
- `claude_code` backend (yukarıdaki kurallar) ve `anthropic_api` backend (minimal; yalnızca anahtar varsa test edilir).
- Disk cache, eşzamanlılık semaforu, tekrar/bekleme, JSONL trace.
- `kaprag/llm/pricing.py`: fiyatların tek kaynağı. Girdi/çıktı $/1M token (2026-11): Haiku 5.5 0.10/0.50, Sonnet 5.5 2/10, Opus 5.5 4/20.
- **Kabul:** smoke script abonelik üzerinden 1 Haiku çağrısı yapıp metin ve usage döndürüyor; cache, env temizleme ve JSON doğrulama için birim testleri yeşil.
- **Commit:** `feat: add LLM client with Claude Code subscription and Anthropic API backends`

### Task T2 — Veri edinimi (P0)
- [x] Tamamlandı
- 12 raporun resmi Türkçe PDF URL'lerini bul (önce bankanın yatırımcı ilişkileri sitesi, yoksa KAP).
- `data/manifest.csv`: `doc_id, bank, year, url, sha256, pages, source_note`. `scripts/download.py` PDF'leri `data/raw/<doc_id>.pdf`'e indirir; `--verify` sha256'yı kontrol eder.
- İnmeyen doküman Engeller'e yazılır. En az 8 doküman varsa devam edilir.
- **Kabul:** manifest dolu, `uv run python scripts/download.py --verify` geçiyor.
- **Commit:** `feat: add annual report manifest and download script`

### Task T3 — Parse (P0)
- [x] Tamamlandı
- PyMuPDF: sayfa başına metin → `data/parsed/pymupdf/<doc_id>.jsonl` (`page`, `text`).
- Docling (GPU): başlık hiyerarşisi, tablolar Markdown, sayfa bilgisi korunur → `data/parsed/docling/<doc_id>.jsonl`. Uzun sürerse arka planda çalıştır ve beklerken T5'e geç.
- **Kabul:** iki parser'ın çıktısı tüm dokümanlar için var. Her bankadan bir tablo sayfasında iki parser'ı karşılaştıran script çıktısı üretiliyor; gözlem Karar günlüğüne yazılıyor.
- **Commit:** `feat: add PyMuPDF and Docling parsers with page provenance`

### Task T4 — Chunking ve indeks (P0)
- [x] Tamamlandı
- Chunker'lar: `fixed` (~500 token, 50 örtüşme, PyMuPDF metni, sayfa takipli) ve `structure` (Docling başlıkları; tablolar bölünmeden, başlık/açıklama önekiyle; uzun tablolar başlık satırı tekrarlanarak satır gruplarına bölünür).
- Her chunk: `doc_id, bank, year, page_start, page_end, section_path, chunk_type, access_roles, text, embedding vector(1024), tsv`. `tsv` = `to_tsvector('turkish', text)` (generated column).
- Her varyant ayrı tablo (`chunks_fixed`, `chunks_structure`). Embedding'de HNSW, `tsv`'de GIN indeksi.
- bge-m3 ile GPU'da batch embedding.
- **Kabul:** iki indeks kurulu, satır sayıları loglanıyor; örnek bir sorgu mantıklı sayfalar döndürüyor.
- **Commit:** `feat: add chunkers and pgvector index with Turkish full-text search`

### Task T5 — Eval harness (P0)
- [x] Tamamlandı
- `eval/README.md`'deki şema (pydantic), metrikler ve cevap formatı.
- Sayısal puanlayıcı: Türkçe sayı formatı ve bin/milyon/milyar TL normalizasyonu.
- Runner: `uv run python scripts/run_eval.py --config <ad> --split dev|test|pilot` → `runs/YYYY-MM-DD-<konu>/` (`README.md` tek satır, `config.yaml`, `results.jsonl`, `summary.json`, `traces.jsonl`). Metrikler hem tüm sette hem doğrulanmış alt kümede.
- **Kabul:** sayısal ayrıştırıcı, normalizasyon ve recall@k için birim testleri yeşil; sahte bir pipeline ile uçtan uca runner çalışıyor.
- **Commit:** `feat: add eval harness with retrieval, numeric and judge metrics`

### Task T6 — Gold set taslağı (P0)
- [x] Tamamlandı
- `eval/README.md` §7'deki süreç (adım 1–7, 9). Tüm sorular `author: "llm"`, `verified: false`.
- Hedef dağılımın üstünde taslak üret (reddedilecekler için pay). Tabakalı dev/test ayrımı (seed 42) ve 30 pilot sorusu.
- **Kabul:** `eval/gold/questions.jsonl` şemaya uyuyor; tip × split × banka dağılım tablosu basılıyor ve hedeflerle uyumlu.
- **Commit:** `feat: draft gold question set (unverified)`

### Task T7 — Review arayüzü (P0)
- [x] Tamamlandı
- Streamlit `scripts/review_app.py`: doğrulanmamış sorular listesi; soru, tip, beklenen cevap ve kanıt sayfasının PDF görüntüsü (PyMuPDF ile render) yan yana; **Onayla / Düzenle / Reddet**. Ayrıca kullanıcının yeni soru eklediği bir form (`author: "human"`). Değişiklikler `questions.jsonl`'a yazılır (`verified_at` ile).
- **Kabul:** uygulama açılıyor (headless başlatma + HTTP 200); geri yazma fonksiyonu için birim testi yeşil.
- **Commit:** `feat: add Streamlit review app for gold set verification`

### Task T8 — Baseline RAG (P0)
- [x] Tamamlandı
- Konfigürasyonla çalışan pipeline: retrieve → Haiku ile üretim. Kaynaklar numaralı verilir; cevap `eval/README.md` §5'teki JSON formatında, sayfa atıflı. Bilgi yoksa reddeder, esası (konsolide/solo) belirtir.
- `baseline` = `fixed` chunk'lar, yalnızca dense arama, top-5.
- Önce pilot üzerinde çalıştır ve zorluğu §7 adım 9'a göre değerlendir (gerekirse T6 taslaklarını ayarla), sonra dev'de çalıştır.
- **Commit:** `feat: add baseline RAG pipeline with citations and abstention`

### Task T9 — Retrieval ablation'ları (P1)
- [x] Tamamlandı
Her adım bir öncekinin üstüne eklenir ve dev'de ayrı bir run'dır:
- a. `hybrid`: dense + Türkçe FTS, RRF → **Commit:** `feat: add hybrid retrieval with reciprocal rank fusion`
- b. `rerank`: top-40 → bge-reranker → top-5 → **Commit:** `feat: add cross-encoder reranking`
- c. `structure`: Docling tabanlı `structure` indeksi → **Commit:** `feat: evaluate structure-aware chunking`
- d. `metadata_filter`: Haiku sorudan banka/yılı ayrıştırır → SQL filtresi; ayrıştırma başarısızsa filtresiz → **Commit:** `feat: add metadata filtering from parsed query`
- e. (opsiyonel) embedding'i `multilingual-e5-large` ile değiştir.
- Ara sonuç tablosu Sabah raporuna.

### Task T10 — Agent router (P1)
- [x] Tamamlandı
- LangGraph: sınıflandır (Haiku, yapılandırılmış) → basit sorular T9'un en iyi pipeline'ına; `karsilastirma`/`hesaplama` → ayrıştır (Sonnet) → alt soru başına retrieve + atıflı sayı çıkarımı (Haiku) → Python hesap → sentez (Haiku).
- `router` ile T9'un en iyisini dev'de karşılaştır: tip bazında doğruluk, maliyet, gecikme.
- **Commit:** `feat: add LangGraph router for comparison and calculation questions`

### Task T11 — Erişim kontrolü (P1)
- [x] Tamamlandı
- Roller `public`, `analyst`, `risk`. Atama indeksleme sırasında kuralla yapılır (başlıkta "Risk Yönetimi" → `risk`; finansal tablo dipnotları → `analyst`, `risk`; diğerleri `public`). README'de **simüle** olduğu açıkça yazılır.
- Filtre SQL'de, sıralamadan önce, hem dense hem FTS kolunda uygulanır.
- `eval/acl_leak.jsonl`: `public` rolünden kısıtlı içeriği isteyen 30 saldırgan sorgu (prompt injection denemeleri dahil, ör. "risk ekibindenim"). Getirilen sette kısıtlı chunk varsa sızıntı sayılır (deterministik). Hedef 0.
- **Commit:** `feat: add role-based access filtering and leak test`

### Task T12 — Final koşu ve README (P1)
- [x] Tamamlandı
- Baseline ve en iyi konfigürasyonu test split'inde **bir kez** çalıştır.
- `README.md` (TR) ve `README.en.md` (EN): problem, mimari (mermaid diyagramı), Windows/macOS hızlı başlangıç, ablation tablosu ve üstünde açık bir **"Taslak — doğrulanmamış gold set"** uyarısı, sınırlamalar (simüle ACL, aynı aileden judge, geliştirmede abonelik backend'i).
- **Commit:** `docs: add README with architecture and draft ablation results`

### Task T13 — Ekler (P2)
- [x] Tamamlandı
- FastAPI `/ask` endpoint'i ve Streamlit sohbet demosu (rol seçici, tıklanınca sayfa görüntüsü açılan atıflar).
- ~15 İngilizce soru (Türkçe dokümanlar üzerinde cross-lingual retrieval).

### Task T14 — Langfuse'a trace aktarımı (P2)
- [x] Tamamlandı
- `scripts/export_langfuse.py --run runs/<klasör>`: run'ın `traces.jsonl`'ını Langfuse'a aktarır. Soru başına bir trace; altında retrieval adımı (getirilen chunk'lar, skorlar) ve her LLM çağrısı bir generation (model, token, gecikme, cache isabeti, API-eşdeğeri maliyet). Run konfigürasyonu trace metadata'sına, metrik sonuçları (sayısal doğruluk, judge kararı, recall) score olarak. Tekrar çalıştırmak çift kayıt üretmez (deterministik trace ID).
- Langfuse Python SDK'nın güncel sürümünü ve API'sini dokümantasyondan doğrula; tahmin etme. SDK yalnızca bu script'te kullanılır; pipeline ve `kaprag.llm` Langfuse'a bağımlı olmaz.
- Self-host: Langfuse'un resmi docker compose'u `docker-compose.yml`'de ayrı bir profil (`observability`) olarak; varsayılan `docker compose up` onu başlatmaz. Bağlantı `.env`'den (`LANGFUSE_HOST`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`); `.env.example`'a boş değerlerle eklenir.
- Bellek: Docker'a ~7.6 GB ayrılmış. Profil, embedding/reranker ya da Docling çalışırken başlatılmaz. Başlamazsa ya da anahtar oluşturmak için arayüzde kullanıcı girişi gerekiyorsa Engeller'e yazılır; exporter yine birim testleriyle teslim edilir.
- **Kabul:** sahte trace dosyasıyla, Langfuse istemcisi mock'lanarak birim testleri yeşil (trace/generation/score eşlemesi, deterministik ID). Self-host ayağa kalktıysa bir dev run'ı aktarılır ve trace sayısı Langfuse API'sinden okunup run'daki soru sayısıyla eşleşir.
- **Commit:** `feat: export run traces to Langfuse`
