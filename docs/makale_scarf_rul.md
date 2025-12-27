# NASA CMAPSS Üzerinde RUL Kestirimi için SCARF Tabanlı Self-Supervised Temsil Öğrenme ve Regresyon

**Tür:** Uygulamalı makine öğrenmesi proje makalesi (yüksek lisans dersi final projesi formatında)

**Yazar:** (Ad Soyad)

**Tarih:** 26 Aralık 2025

---

## Özet

Bu çalışmada NASA’nın CMAPSS turbofan motor veri seti üzerinde Remaining Useful Life (RUL) kestirimi problemi ele alınmıştır. Proje, çok değişkenli zaman serilerini sabit boyutlu tablosal vektörlere dönüştüren kayan pencere (sliding window) yaklaşımını ve bu vektörler üzerinde SCARF benzeri self-supervised kontrastif ön-eğitimi birleştirir. Ön-eğitim, etiket gerektirmeden sensörler arası ilişkileri temsil uzayına taşımayı hedefler; ardından RUL regresyonu için aşamalı fine-tuning (önce sadece head, sonra encoder+head) uygulanır. Değerlendirme, CMAPSS’in resmi test protokolüne uygun şekilde her test motoru için “son pencere” üzerinden yapılır ve RMSE/MAE metrikleri raporlanır. Mevcut deneylerde SCARF tabanlı yaklaşım bazı koşullarda supervised baseline ile benzer sonuçlar verirken, belirgin ve tutarlı bir üstünlük göstermemiştir. Bu bulgu, flatten tabanlı MLP encoder seçimi, corruption tasarımı ve subset’lerin çok koşullu doğasıyla ilişkilendirilerek tartışılmış; ablation, encoder mimarisi değişimi ve seed-ortalaması gibi iyileştirme önerileri sunulmuştur.

**Anahtar Kelimeler:** RUL kestirimi, CMAPSS, self-supervised learning, SCARF, kontrastif öğrenme, InfoNCE, zaman serisi, prognostics

---

## 1. Giriş

Kalan faydalı ömür (RUL) kestirimi, kestirimci bakım (predictive maintenance) ve arıza öncesi karar destek süreçlerinde kritik bir problem sınıfıdır. Jet motoru gibi yüksek maliyetli sistemlerde arızaya kadar veri toplamak pahalı ve risklidir; bu nedenle etiketli veri (özellikle arızaya yakın bölgede) sınırlı olabilir. Bu bağlamda self-supervised temsil öğrenme, etiket gereksinimini azaltarak sensör verilerinden genel amaçlı bir temsil elde etmeyi hedefler.

Bu projede amaç, CMAPSS veri setinde RUL regresyonu için:

- Zaman serisini tablosal bir forma dönüştürmek,
- SCARF tarzı self-supervised kontrastif ön-eğitim ile encoder temsili öğrenmek,
- Az etiketli senaryolarda dahi fine-tuning ile performansı koruyabilmektir.

---

## 2. Veri Seti: NASA CMAPSS

CMAPSS veri seti farklı çalışma koşulları ve arıza modları içeren alt kümelerden oluşur:

- **FD001:** 100 train / 100 test, 1 koşul, 1 arıza modu
- **FD002:** 260 train / 259 test, 6 koşul, 1 arıza modu
- **FD003:** 100 train / 100 test, 1 koşul, 2 arıza modu
- **FD004:** 248 train / 249 test, 6 koşul, 2 arıza modu

Her satır bir çevrimde (cycle) alınan snapshot’tır. Kolonlar: motor kimliği (unit), zaman (cycle), 3 operasyon ayarı (setting) ve sensör ölçümleridir.

---

## 3. Deney Protokolü ve Veri Sızıntısı (Leakage) Önlemleri

RUL gibi zaman bağımlı problemlerde yanlış split/ölçekleme, yapay biçimde yüksek metriklere yol açabilir. Bu repo şu temel önlemleri alır:

1. **Unit-bazlı train/val split:** Train/val ayrımı motor (unit) bazında yapılır.

2. **Train-only scaling:** MinMaxScaler yalnızca train unit’lerinin verisine fit edilir; val/test’e sadece transform uygulanır.

3. **Sabit kolonların elenmesi:** Train subset’te varyansı 0 olan kolonlar bilgi taşımadığı için drop edilir.

Bu protokol, pencere bazlı karışma ve ölçekleme kaynaklı leakage’ı azaltmayı hedefler.

---

## 4. Zaman Serisinden Tabloya: Kayan Pencere Temsili

### 4.1 Pencere Oluşturma

Her motorun çok değişkenli zaman serisi, pencere boyutu $w$ olacak şekilde parçalara ayrılır:

- Pencere boyutu: $w=50$
- Her pencere: $w \times d$ (d: seçilen feature sayısı)
- Model girdisi: pencere matrisi flatten edilerek tek bir vektöre dönüştürülür: $x \in \mathbb{R}^{w\cdot d}$

Bu yaklaşım SCARF gibi tablosal veri odaklı yöntemlerle uyum sağlar; ancak zaman sıralı yapıyı (özellikle lokal örüntüleri) açıkça modellemez.

### 4.2 RUL Etiketleme ve Cap

Train içinde pencere son çevrimine göre RUL şöyle tanımlanır:

$$
\mathrm{RUL} = \max\_\text{cycle}(\text{unit}) - \mathrm{end\_cycle}(\text{window})
$$

RUL, çok büyük değerlerin etkisini sınırlamak için $\mathrm{cap}=125$ ile kırpılır.

---

## 5. Yöntem

Proje üç eğitim modu uygular:

1. **supervised_only:** SCARF ön-eğitim olmadan encoder+regression head uçtan uca eğitilir.

2. **scarf_head_only:** SCARF ile encoder ön-eğitilir; fine-tuning sırasında encoder dondurulup yalnızca regression head eğitilir.

3. **scarf_full:** SCARF ile encoder ön-eğitilir; fine-tuning iki fazlıdır: (i) head-only fazı, (ii) encoder+head birlikte fazı.

### 5.1 Encoder ve Regresyon Head

Encoder, MLP tabanlı bir ağdır:

- Gizli boyut: 256
- Embedding boyutu: 64
- Dropout: 0.1

Regresyon modeli, encoder’ın embedding çıktısı üzerine küçük bir MLP head ekleyerek tek skalar RUL üretir.

### 5.2 SCARF Corruption (Augmentation)

SCARF tarzı bozulma, feature seviyesinde batch içi permütasyon ile yapılır:

- Maske: her feature için Bernoulli($p=\text{corruption\_rate}$)
- Maskelenen feature değerleri aynı batch’ten rastgele permüte edilmiş örneklerden kopyalanır.

Bu augmentation, sensör gürültüsü/eksikliği gibi varyasyonlara karşı daha dayanıklı temsil hedefler.

### 5.3 Kontrastif Kayıp: InfoNCE

Encoder+projector çıktıları normalize edilerek $z$ ve bozulmuş versiyon $\tilde{z}$ elde edilir. Logits:

$$
\mathrm{logits} = \frac{z \tilde{z}^\top}{T}
$$

Burada $T$ sıcaklık (temperature) hiperparametresidir. Her örnek için doğru sınıf kendi pozitif eşidir; kayıp cross-entropy ile hesaplanır.

---

## 6. Eğitim Detayları

### 6.1 Ön-eğitim (Pretraining)

- Optimizer: Adam
- Öğrenme oranı (pretrain): 1e-3
- Weight decay: 1e-5
- Batch size: 256
- Epoch: deney senaryosuna göre (ör. 5 veya 10)

Not: Grid aramasında kullanılan uygulamada, label*fraction < 1.0 iken pretraining’in “tam unlabeled havuz” yerine subsample edilmiş $X*{tr}$ üzerinde çalışması mümkündür (grid döngüsünde pretrain fonksiyonuna $X_{tr}$ verilmekte). Bu, self-supervised faydayı azaltabilecek bir uygulama farkıdır ve gelecekte düzeltme/ablasyon gerektirir.

### 6.2 Fine-tuning

**Faz 1 (head-only):** encoder dondurulur, head LR=1e-3

**Faz 2 (full):** encoder açılır; encoder LR=1e-4, head LR=1e-3

Kayıp: MSE (alternatif: Huber / SmoothL1Loss)

---

## 7. Değerlendirme

### 7.1 Validation

Train dosyasında unit-split ile ayrılan val pencereleri üzerinde değerlendirme yapılır.

### 7.2 Official Test Protokolü

CMAPSS test değerlendirmesi şu şekilde uygulanır:

- Her test motoru için yalnızca **son pencere** alınır (last window per engine).
- Gerçek RUL değerleri `RUL_<subset>.txt` dosyasından alınır.
- Metrikler: RMSE ve MAE.

---

## 8. Deneyler ve Bulgular

### 8.1 Mevcut En İyi Test Sonuçları (FD001–FD004)

Aşağıdaki değerler doğrudan proje çıktısı olan metrics.json dosyalarından alınmıştır (26 Aralık 2025 itibarıyla):

| Subset | RMSE_test | MAE_test |
| ------ | --------: | -------: |
| FD001  |   17.2881 |  12.8802 |
| FD002  |   29.9849 |  25.7303 |
| FD003  |   54.8264 |  49.1505 |
| FD004  |   31.9229 |  27.2276 |

Kaynaklar (tek tek):

- FD001: evaluation_fd001_test/metrics.json (rmse_test=17.288070678710938, mae_test=12.88017463684082)
- FD002: evaluation_fd002_test/metrics.json (rmse_test=29.984882354736328, mae_test=25.730348587036133)
- FD003: evaluation_fd003_test/metrics.json (rmse_test=54.82643127441406, mae_test=49.15047073364258)
- FD004: evaluation_fd004_test/metrics.json (rmse_test=31.92293930053711, mae_test=27.227609634399414)

Gözlem: FD003 belirgin şekilde daha zordur (çok daha yüksek hata). Çok koşullu subset’lerde (FD002/FD004) hata da artmaktadır.

### 8.2 Suite Deneyi (FD001 örneği)

evaluation_suite/summary.csv içinde FD001 için üç mod raporlanmıştır (tam değerler):

- supervised_only: rmse_test=15.325263977050781, mae_test=11.241100311279297
- scarf_full: rmse_test=17.288070678710938, mae_test=12.88017463684082
- scarf_head_only: rmse_test=66.1172866821289, mae_test=55.35366439819336

Kaynak: evaluation_suite/summary.csv (FD001 satırları)

Yorum: Head-only fine-tuning, bu konfigürasyonda RUL hedefi için yeterli olmamış; encoder’ın açıldığı full fine-tuning daha mantıklı bir stratejidir. Ancak supervised baseline bu koşullarda daha iyi sonuç vermiştir.

### 8.3 Label-Fraction (Etiket Verimliliği) Deneyi

evaluation_label_sweep/summary.csv üzerinde iki subset için label_fraction (etiket oranı) taraması yapılmıştır.

**FD004:**

- lf=0.05

  - supervised_only: rmse_test=43.80696487426758, mae_test=38.01313400268555
  - scarf_full: rmse_test=48.639549255371094, mae_test=39.95475387573242

- lf=0.10

  - supervised_only: rmse_test=44.18254089355469, mae_test=38.011924743652344
  - scarf_full: rmse_test=44.458900451660156, mae_test=38.424583435058594

- lf=0.25

  - supervised_only: rmse_test=44.00531005859375, mae_test=37.73841857910156
  - scarf_full: rmse_test=43.766029357910156, mae_test=37.7479133605957

- lf=1.00
  - supervised_only: rmse_test=29.201051712036133, mae_test=23.809728622436523
  - scarf_full: rmse_test=36.611534118652344, mae_test=30.871118545532227

Kaynak: evaluation_label_sweep/summary.csv (FD004 satırları)

Gözlem: Orta etiket oranlarında (0.25 civarı) yakın performans görülse de, yüksek etiket oranında supervised daha iyi görünmektedir.

### 8.4 Hiperparametre Taraması (FD002/FD004, lf=0.1)

Grid: corruption_rate ∈ {0.3, 0.6}, temperature ∈ {0.05, 0.1}, lr_encoder ∈ {5e-5, 1e-4}

Örnek (FD004, lf=0.1) — tam değerler:

- supervised_only: rmse_test=44.177574157714844, mae_test=37.95985412597656
- scarf_full_best: rmse_test=44.178672790527344, mae_test=38.018524169921875

Kaynak: evaluation_hparam_sweep_fd004/summary.csv (FD004, label_fraction=0.1 satırları)

Bu sonuç, seçilen mimari/augmentation ile self-supervised pretraining’in test performansına net bir katkı vermediğini göstermektedir.

---

## 9. Tartışma

### 9.1 Neden SCARF Kazancı Sınırlı Olabilir?

- **Flatten + MLP encoder:** Zaman içi yapıyı açıkça modellemediği için temsil kalitesi sınırlanabilir.
- **Corruption tasarımı:** Feature permütasyonu bazı subsetlerde RUL ile ilişkili kritik sinyali bozabilir.
- **Multi-condition zorluğu:** FD002/FD004’te çalışma koşulları çeşitlidir; model koşul değişimini arızadan ayırmakta zorlanabilir.
- **Hedef uyumsuzluğu:** Kontrastif hedef (benzerlik ayrıştırma) ile RUL regresyon hedefi her zaman aynı yönde optimize olmayabilir.

### 9.2 Geçerlilik Tehditleri

- **Seed hassasiyeti:** Tek seed ile raporlama yanıltıcı olabilir; ortalama±std önerilir.
- **Hyperparametre kapsamı:** Grid sınırlı; daha geniş arama veya Bayes optimizasyonu anlamlı olabilir.
- **Pretraining havuzu farkı:** Grid senaryosunda pretraining’in subsample üzerinde olması olası bir metodoloji farkıdır.

---

## 10. Sonuç

Bu proje, CMAPSS üzerinde RUL kestirimi için self-supervised SCARF ön-eğitimini ve aşamalı fine-tuning’i uçtan uca çalışan bir pipeline içinde sunmaktadır. Deney protokolü leakage risklerini azaltacak şekilde unit-split ve train-only scaling prensiplerine dayanır. Mevcut deneylerde SCARF yaklaşımı supervised baseline’a kıyasla tutarlı iyileşme göstermemiş; buna rağmen proje, self-supervised öğrenmenin RUL problemlerine uygulanması, ablation/sweep altyapısı ve resmi test protokolüyle değerlendirme gibi güçlü “yüksek lisans seviyesi” bileşenler içermektedir.

---

## 11. Gelecek Çalışmalar (Projeyi Güçlendirme)

- **Encoder mimarisi:** MLP yerine 1D-CNN / LSTM / Transformer tabanlı encoder
- **Corruption ablation:** Feature permütasyonu yerine masking/noise/temporal jitter
- **Seed ortalaması:** 5–10 farklı seed ile mean±std raporlama
- **Ablation:** window_size, rul_cap, loss (MSE vs Huber), temperature ve corruption oranları
- **Koşul ayrıştırma:** operating setting’leri ayrı işlemek veya domain-adaptation benzeri yaklaşımlar

---

## 12. Reprodüksiyon (Çalıştırma)

Örnek komutlar:

- Eğitim (SCARF full):

  - `python experiment_window_size.py --dataset-root Datasets --subset FD004 --window-size 50 --rul-cap 125 --method scarf_full --out-dir artifacts_fd004_run`

- Official test değerlendirme:

  - `python evaluate_best.py --dataset-root Datasets --subset FD004 --eval test --window-size 50 --rul-cap 125 --best-dir <RUN_DIR> --out-dir evaluation_fd004_test`

- Hparam sweep:
  - `python run_hparam_sweep.py --subset FD004 --label-fraction 0.1 --grid-corruption 0.3,0.6 --grid-temperature 0.05,0.1 --grid-lr-encoder 5e-05,1e-4 --pretrain-epochs 10 --ft-head-epochs 5 --ft-full-epochs 10 --train-root artifacts_hparam_sweep_fd004 --eval-root evaluation_hparam_sweep_fd004 --summary-csv evaluation_hparam_sweep_fd004/summary.csv`

---

## Ek A: Repo Organizasyonu (Kısa)

- Eğitim ve temel pipeline: `experiment_window_size.py`
- Model değerlendirme: `evaluate_best.py`
- Çoklu subset/method koşuları: `run_suite.py`
- Hparam sweep yardımcı: `run_hparam_sweep.py`
- Etiket oranı grafiği: `plot_label_sweep.py`
