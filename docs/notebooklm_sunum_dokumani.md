# NotebookLM Slayt Dokümanı — NASA CMAPSS RUL Kestirimi (SCARF Adaptasyonu)

Bu doküman NotebookLM’ye yüklenmek üzere hazırlanmıştır. Amaç: proje hikayesini, yöntemi, deney protokolünü ve elde edilen sonuçları “slayt üretilebilir” bir formatta tek yerde toplamak.

---

## 1) Proje Özeti (1 slayt)

**Başlık:** NASA CMAPSS Turbofan Veri Setinde Remaining Useful Life (RUL) Kestirimi — SCARF (Self-Supervised) + Regresyon

**Problem:** Test motorları sistem arızasından önce kesiliyor; hedef son gözlemden itibaren kalan çevrim sayısını (RUL) tahmin etmek.

**Ana fikir:**

- Zaman serisini “kayan pencere + düzleştirme” ile tablosal bir vektöre çevir.
- Etiketsiz pencerelerde SCARF ile temsili (encoder) self-supervised öğren.
- Az etiketli durumda dahi RUL regresyonu için fine-tuning yap.

**Neden önemli:**

- Etiket maliyeti yüksek (arızaya kadar gerçek RUL etiketi toplamak zor).
- Self-supervised temsil öğrenme ile veri verimliliği hedeflenir.

---

## 2) Veri Seti (1 slayt)

**NASA CMAPSS alt kümeleri:**

- FD001: 100 train / 100 test, 1 koşul, 1 arıza modu
- FD002: 260 train / 259 test, 6 koşul, 1 arıza modu
- FD003: 100 train / 100 test, 1 koşul, 2 arıza modu
- FD004: 248 train / 249 test, 6 koşul, 2 arıza modu

**Ham format:** Her satır bir “cycle” snapshot; kolonlar: unit id, cycle, 3 operating setting + sensörler.

---

## 3) Deney Protokolü (1 slayt)

Bu proje için kritik metodolojik nokta “data leakage” önlemi:

- **Split unit-bazlı:** Train/val ayrımı motor (unit) bazında yapılıyor.
- **Scaler sadece train unit’lerinde fit:** Val/test’e sadece transform uygulanıyor.
- **Sabit kolonlar:** Train subset’te varyansı 0 olan kolonlar drop ediliyor.

Bu sayede pencere bazlı karışma ve ölçekleme kaynaklı leakage azaltılıyor.

---

## 4) Özellik Mühendisliği: Zaman Serisinden Tabloya (1 slayt)

**Kayan pencere (sliding window):**

- Pencere boyutu: `window_size = 50`
- Her pencere: `window_size × feature_count` matrisi
- Model girdisi: pencere matrisi **flatten** edilerek tek bir uzun vektör

**RUL label:**

- Train içinde: her pencerenin `end_cycle` değeri üzerinden
  \(\mathrm{RUL} = \max_cycle(unit) - end_cycle\)
- RUL cap: `125` (çok büyük RUL’ların kırpılması)

---

## 5) Model: Encoder + Regressor (1 slayt)

**Encoder (MLP):**

- Input: flatten edilmiş pencere vektörü
- Embedding: `emb_dim = 64`
- Projector: InfoNCE için normalize edilmiş `proj_dim = 64`

**Regressor:**

- Encoder çıktısı (h) → küçük MLP head → 1 skalar RUL

---

## 6) SCARF Self-Supervised Ön-Eğitim (2 slayt)

### 6.1 Augmentation / Corruption

SCARF tarzı corruption: feature seviyesinde “batch içinden permütasyonla” bozma.

- Maske: her feature için \(p=\texttt{corruption_rate}\)
- Maskelenen feature’lar başka örnekten kopyalanıyor.

Bu, sensör ölçüm gürültüsü/eksikliği benzeri bir robustluk hedefler.

### 6.2 Kontrastif Kayıp (InfoNCE)

- Orijinal pencere: \(x\)
- Bozulmuş pencere: \(\tilde{x}\)
- Projeksiyon uzayı: \(z, \tilde{z}\) (L2 normalize)

Logits:
\[ \mathrm{logits} = \frac{z \tilde{z}^T}{T} \]

Kayıp: her örnek için “kendi pozitif eşini” sınıflandırma (cross-entropy).

---

## 7) Fine-tuning Stratejileri (1 slayt)

Bu repoda 3 mod var:

1. **supervised_only**

- Ön-eğitim yok
- Encoder + head birlikte eğitilir

2. **scarf_head_only**

- SCARF ile ön-eğitim
- Sadece head eğitimi (encoder dondurulabilir; bu deneyde zayıf kaldı)

3. **scarf_full**

- SCARF ile ön-eğitim
- Aşamalı fine-tuning: head phase + full phase (encoder + head)

Ek deney: **label_fraction** ile etiketli veri miktarını düşürüp veri verimliliği gözlenir.

---

## 8) Değerlendirme (1 slayt)

İki değerlendirme modu:

- **Validation:** train dosyasında unit-split ile oluşturulan val pencereleri
- **Official test protokolü:** her test motoru için **son pencere** alınır (last window per engine) ve `RUL_<subset>.txt` ile karşılaştırılır.

Metrikler: RMSE ve MAE.

---

## 9) Mevcut En İyi Test Sonuçları (1 slayt)

Aşağıdaki sayılar, bu çalışma alanındaki `evaluation_fd00X_test/metrics.json` dosyalarından alınmıştır:

| Subset | RMSE_test | MAE_test |
| ------ | --------: | -------: |
| FD001  |   17.2881 |  12.8802 |
| FD002  |   29.9849 |  25.7303 |
| FD003  |   54.8264 |  49.1505 |
| FD004  |   31.9229 |  27.2276 |

Not: Bu sayılar kullanılan koşullar/pencereler ve eğitim ayarlarına göre değişebilir.

---

## 10) Label-Fraction Deneyi (1–2 slayt)

Bu çalışmada `evaluation_label_sweep/summary.csv` sonuçları:

**FD001 (kısa eğitim: pre2/h2/f4):**

- Çok düşük etiket oranlarında (0.01–0.1) hem supervised hem SCARF benzer ve yüksek hata
- Etiket arttıkça hata düşüyor; bu koşulda supervised genellikle daha iyi

**FD004 (pre5/h3/f6):**

- `lf=0.25` civarında supervised ve SCARF çok yakın
- `lf=1.0`’da supervised daha iyi

Slayta öneri: “etiket azaldıkça RMSE artıyor; SCARF beklenen kazanımı bu konfigürasyonda net göstermedi” şeklinde dürüst bir çıkarım.

---

## 11) Hparam Sweep (FD002/FD004, lf=0.1) (1 slayt)

`evaluation_hparam_sweep_fd002/summary.csv` ve `evaluation_hparam_sweep_fd004/summary.csv` özet:

- Sweep grid: `corruption_rate ∈ {0.3, 0.6}`, `temperature ∈ {0.05, 0.1}`, `lr_encoder ∈ {5e-5, 1e-4}`
- Seçim kriteri: **best_val_rmse**

Örnek (FD004, lf=0.1):

- Best SCARF tag: `ws50_pre10_h5_f10_cr0p6_t0p05_lre0p0001_lrh0p001_lf0p1`

Not: Bu sweep sonuçlarında supervised_only ile scarf_full_best test hataları birbirine çok yakın.

---

## 12) Neden SCARF burada sınırlı kazanç verdi? (1 slayt)

Sunumda “olgun” görünmek için kısa ve teknik açıklama:

- Corruption mekanizması (feature permütasyonu) bazı subsetlerde RUL ile ilişkili sinyali zayıflatıyor olabilir.
- Encoder (MLP) zaman içi yapıyı tamamen flatten ederek kaybediyor; daha uygun encoder (1D-CNN / LSTM / Transformer) temsil kalitesini artırabilir.
- Kontrastif hedef ile RUL hedefi her zaman aynı yönü optimize etmiyor; özellikle multi-condition subsetlerde (FD002/FD004) koşul ayrıştırma zor.

---

## 13) Projeyi “Yüksek Lisans Dersi Final Projesi” Olarak Güçlendirme (1 slayt)

Eğer hocan “katkı nerede?” diye sorarsa, şu net geliştirmeler sunulabilir:

- **Ablation:** window_size, rul_cap, loss (MSE vs Huber), corruption_rate, temperature etkisi
- **Leakage doğrulaması:** unit-split vs window-split karşılaştırması (leakage’ın metriklere etkisi)
- **Model karşılaştırması:** MLP vs 1D-CNN/LSTM encoder (aynı protokol, aynı split)
- **İstatistiksel sağlamlık:** farklı seed’lerle ortalama±std raporlama
- **Hata analizi:** düşük/orta/yüksek RUL bölgelerinde hata dağılımı

---

## 14) Reprodüksiyon: Çalıştırma Komutları (1 slayt)

Aşağıdaki örnekler bu repodaki script isimleriyle uyumludur:

- Tek subset eğitim (SCARF full):

  - `python experiment_window_size.py --dataset-root Datasets --subset FD004 --window-size 50 --rul-cap 125 --method scarf_full --out-dir artifacts_fd004_run`

- En iyi checkpoint’i official test ile değerlendir:

  - `python evaluate_best.py --dataset-root Datasets --subset FD004 --eval test --window-size 50 --rul-cap 125 --best-dir <RUN_DIR> --out-dir evaluation_fd004_test`

- Hparam sweep:
  - `python run_hparam_sweep.py --subset FD004 --label-fraction 0.1 --grid-corruption 0.3,0.6 --grid-temperature 0.05,0.1 --grid-lr-encoder 5e-05,1e-4 --pretrain-epochs 10 --ft-head-epochs 5 --ft-full-epochs 10 --train-root artifacts_hparam_sweep_fd004 --eval-root evaluation_hparam_sweep_fd004 --summary-csv evaluation_hparam_sweep_fd004/summary.csv`

---

## 15) Ek: Sunumda Gösterilecek Artefaktlar (1 slayt)

Bu klasörler “kanıt” olarak gösterilebilir:

- `evaluation_fd00X_test/` → `metrics.json`, `test_predictions_*.csv`, `test_true_vs_pred_*.png`
- `evaluation_hparam_sweep_*/summary.csv` → sweep özeti
- `artifacts_*/` → eğitim koşulları ve checkpoint’ler

---

## Slayt üretimi için prompt önerisi (NotebookLM’ye)

"Bu dokümanı kullanarak 12–15 slaytlık bir sunum üret. Her slaytta başlık + 3–6 madde olsun. Sonuç slaydında FD001–FD004 test RMSE/MAE tablosu yer alsın. Deney protokolü slaydında unit-split ve train-only scaling ile leakage önlemini özellikle vurgula. Son slaytta geliştirme önerileri ver."
