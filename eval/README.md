# Eval seti tasarımı

> Bu doküman eval setinin tasarımının sahibidir: soru tipleri, şema, puanlama kuralları, metrikler ve gold set üretim süreci. Görev durumu ve uygulama planı `PLAN.md`'de yaşar.

## Amaç

Her mimari değişikliğin (hibrit arama, reranker, chunking, agent router…) kaliteye, maliyete ve gecikmeye etkisini **ölçerek** göstermek. Projenin vitrini bu eval'den çıkan ablation tablosudur.

## 1. Soru tipleri ve hedef dağılım (~180 soru)

| Tip (`type`) | Hedef | Örnek | Ne ölçer |
|---|---|---|---|
| `olgusal` | 30 | "Akbank'ın 2025 sonunda kaç şubesi var?" | Temel retrieval |
| `tablo` | 40 | "İş Bankası'nın 2024 konsolide net kârı ne kadar?" | Tablo parse, birim |
| `hesaplama` | 20 | "Yapı Kredi'nin net kârı 2024'ten 2025'e yüzde kaç değişti?" | Doğru sayıları çekip hesaplamak |
| `karsilastirma` | 30 | "2025'te sermaye yeterlilik oranı en yüksek banka hangisi?" | Çok doküman, metadata filtresi, agent |
| `sentez` | 20 | "Garanti 2025 raporunda sürdürülebilirlik hedefleri neler?" | Birden çok chunk'ı birleştirmek |
| `cevapsiz` | 25 | Korpusta olmayan banka, yanlış öncül, raporda olmayan metrik | Halüsinasyon / reddetme |
| `belirsiz` | 15 | Konsolide/solo ya da banka belirtilmemiş | Varsayımı açıkça söylemek / netleştirme istemek |

Cevapsız soruların payı (~%14) bilinçli: bu oran yoksa sistem "her şeye cevap ver" yönünde ödüllendirilir.

## 2. Dev / test ayrımı

- **dev (60):** Her denemede çalışır, iyileştirmeler buna bakılarak yapılır.
- **test (120):** Yalnızca önemli versiyonlarda (baseline ve final aday) çalışır. README'deki tablo buradan gelir.
- Ayrım tipe ve bankaya göre tabakalı, sabit seed (42). Kullanıcı yeni soru ekledikçe ayrım aynı kuralla güncellenir; mevcut atamalar değişmez.
- `pilot: true` işaretli 30 soru (tabakalı) zorluk kalibrasyonu içindir.

## 3. Soru şeması (`eval/gold/questions.jsonl`, satır başına bir JSON)

```json
{"id": "q042", "split": "test", "pilot": false, "type": "hesaplama",
 "question": "Yapı Kredi'nin konsolide net kârı 2024'ten 2025'e yüzde kaç değişti?",
 "banks": ["yapikredi"], "years": [2024, 2025], "basis": "konsolide",
 "expected_behavior": "answer",
 "expected": {"answer": "…", "value": null, "unit": "%", "tolerance": {"type": "abs", "amount": 0.5}},
 "evidence": [{"doc": "yapikredi-2025", "page": null}],
 "key_points": [],
 "derived_from": [],
 "author": "llm", "llm_checked": true, "verified": false, "verified_at": null,
 "audit": {"status": "passed", "by": "agent", "at": "2026-11-18", "note": ""}, "notes": ""}
```

| Alan | Açıklama |
|---|---|
| `expected_behavior` | `answer` · `abstain` (bilgi korpusta yok) · `clarify` (soru belirsiz, netleştirme iste) · `answer_with_assumption` (varsayımı söyleyerek cevapla) |
| `expected.value` / `unit` | Sayısal sorularda normalize değer. Para **TL** cinsinden tam sayı (bin/milyon/milyar çarpanı uygulanmış), oran **%** cinsinden, adet tam sayı. |
| `expected.tolerance` | Para: `{"type": "rel", "amount": 0.01}` (yuvarlanmış "45,2 milyar TL" gibi ifadeler için). Oran: `{"type": "abs", "amount": 0.2}` puan. Adet: `{"type": "abs", "amount": 0}`. |
| `evidence` | **Sayfa düzeyinde** gold kanıt (`doc` + 1 tabanlı `page`). Chunk ID'si değil; chunking deneyler boyunca değişir, sayfa değişmez. Bilgi birden çok sayfada geçiyorsa hepsi yazılır. |
| `key_points` | `sentez` sorularında cevabın kapsaması gereken maddeler. |
| `derived_from` | Tekil sorulardan birleştirilen `karsilastirma`/`hesaplama` sorularının kaynak ID'leri. |
| `author` | `llm` ya da `human`. |
| `llm_checked` | Taslak, ikinci bir LLM çağrısıyla sayfaya karşı kontrol edildi mi. Doğrulama **sayılmaz**. |
| `verified` | Yalnızca kullanıcı, review arayüzünden PDF sayfasına bakarak `true` yapar. Agent asla yapmaz. |
| `audit` | `null` (denetlenmedi) ya da bağımsız bir agent'ın sayfa görüntüsüne karşı denetimi: `{"status", "by": "agent", "at": "YYYY-MM-DD", "note"}`. `passed` sorun bulunmadı · `fixed` düzeltmeleri uygulandı · `flagged` insan kararı gerekiyor, satır değiştirilmedi. Yalnızca `scripts/apply_audit.py` yazar; review arayüzü salt okunur gösterir. **Doğrulama sayılmaz**, `verified`'ı değiştirmez. Denetlenmiş alt küme = `passed` + `fixed`. |

## 4. Puanlama kuralları

- **Hangi rakam esas?** Tek yıllık değerde o yılın raporu esastır. Yıllar arası değişimde en güncel raporun karşılaştırmalı sütunu kullanılır (önceki yıl rakamları sonradan yeniden düzenlenmiş olabilir).
- **Konsolide / solo:** Soru belirtmiyorsa sistem konsolide rakamı kullanır ve bunu cevapta söyler (`answer_with_assumption`). Banka ya da yıl belirtilmemişse beklenen davranış `clarify`.
- **Korpus dışı yıllar dikkat:** 2023 raporu 2022'nin karşılaştırmalı rakamlarını, bazı raporlar 5 yıllık özet tabloları içerir. Bir yılın korpusta olmadığı varsayılmaz; `cevapsiz` sorular korpusta aranarak doğrulanır.

## 5. Sistemin cevap formatı

Pipeline her soruya yapılandırılmış çıktı döner; sayısal puanlama böylece deterministik olur:

```json
{"answer": "…", "value": 12345678000, "unit": "TL", "basis": "konsolide",
 "citations": [{"doc": "akbank-2024", "page": 87}], "abstained": false, "clarification": null}
```

## 6. Metrikler

**Retrieval** (cevaptan bağımsız, sayfa düzeyinde):
- `page_recall@5`, `page_recall@10`: gold sayfalardan en az biri ilk k sonuçta mı?
- `all_evidence_recall@10`: çok kanıtlı sorularda gold sayfaların hepsi getirildi mi?
- `mrr@10`

**Cevap:**
- **Sayısal doğruluk** (`tablo`, `hesaplama`, sayısal `karsilastirma`): `value` ve `unit` toleransla karşılaştırılır. LLM judge yok; Türkçe sayı formatı (`1.234.567,8`) ve "bin/milyon/milyar TL" normalizasyonu birim testleriyle güvenceye alınır.
- **Metin doğruluğu** (`olgusal`, sayısal olmayan `karsilastirma`): LLM judge (`claude-sonnet-5-5`) referans cevapla karşılaştırır → doğru / yanlış. Judge, kullanıcının elle etiketlediği 50 cevapla karşılaştırılarak kalibre edilir ve uyum oranı raporlanır. Sınırlama: judge ile üretici aynı model ailesinden.
- **Key-point kapsama** (`sentez`): kapsanan madde oranı (judge).
- **Kaynak doğruluğu:** atıf yapılan sayfaların gold sayfalarla kesişimi (precision / recall).
- **Davranış doğruluğu:** `abstain` sorularında doğru reddetme oranı; `answer` sorularında yanlış reddetme oranı; `clarify` ve `answer_with_assumption` sorularında beklenen davranış gösterildi mi. Doğru reddetme ile yanlış reddetme her zaman birlikte raporlanır.
- **Operasyonel:** soru başına API-eşdeğeri maliyet (token × liste fiyatı), p50/p95 gecikme (cache isabetleri hariç).

**Raporlama:** Her metrik hem tüm sette hem `verified: true` alt kümesinde hesaplanır. README ve CV'ye yalnızca doğrulanmış alt kümenin sayıları girer. Metrikler ayrıca agent denetiminden geçmiş alt kümede (`audited`: `audit.status` `passed`/`fixed`) hesaplanır; agent denetimi insan doğrulaması değildir.

## 7. Gold set üretimi

1. **Sayfa örnekleme:** Her dokümandan metin ağırlıklı ve tablo ağırlıklı sayfalar tabakalı örneklenir.
2. **Taslak:** `claude-sonnet-5-5` sayfadan, tipe uygun soru + beklenen cevap + değer/birim/esas + kanıt sayfası üretir. Sorular **bir analistin soracağı dille** yazılır; sayfadaki ifadeler kopyalanmaz ("dönem net kârı" yerine "ne kadar kâr etti"). Aksi halde retrieval olduğundan iyi görünür.
3. **LLM kontrolü:** İkinci bir çağrı cevabın sayfada gerçekten bulunduğunu kontrol eder; geçmeyen taslak atılır, geçen `llm_checked: true` olur.
4. **Birleştirme:** `karsilastirma` ve bazı `hesaplama` soruları, aynı metriğin farklı banka/yıllardaki tekil sorularından üretilir (`derived_from`). Kaynakları doğrulanınca bunlar da doğrulanmış sayılabilir.
5. **Cevapsız:** Korpus dışı bankalar (Ziraat, Halkbank, QNB…), yanlış öncül ("…zararı ne kadardı?" kâr eden bir banka için), raporda bulunmayan makul metrikler. Her biri korpusta hibrit aramayla kontrol edilir.
6. **Belirsiz:** Mevcut tekil sorulardan banka, yıl ya da esas bilgisi çıkarılarak üretilir.
7. **Fazla üretim:** Reddedilecekler hesaba katılarak hedeften fazla taslak üretilir.
8. **İnsan doğrulaması:** Kullanıcı review arayüzünde soruyu ve PDF sayfasını yan yana görüp onaylar, düzeltir ya da reddeder (soru başına ~1–2 dk). Hedef: final setin en az %35'i insan yazımı soru (`author: human`); reddedilen taslakların yerini kullanıcı soruları alır.
9. **Pilot:** Önce 30 pilot soruyla baseline çalıştırılır. Baseline doğruluğu %85'in üstündeyse set fazla kolay, %15'in altındaysa fazla zordur; taslak üretimi buna göre ayarlanır.
