# Panduan Penyebaran & Manajemen Aset Model Biner

Dokumen ini menjelaskan urutan eksekusi lengkap sistem dari awal, metodologi pelatihan model, anatomi file biner, dan best practices untuk deployment dan maintenance.

---

## 1. Alur Eksekusi Lengkap Pipeline (End-to-End Workflow)

### Fase Persiapan: Setup Environment

Sebelum menjalankan sistem, pastikan:

1. **Python 3.7+** terinstall
2. **Pustaka yang diperlukan** sudah diinstall:
   ```powershell
   pip install scikit-learn python-crfsuite numpy pandas
   ```

3. **Struktur direktori** sudah sesuai:
   ```
   d:\Project\Axolotl\
   ├── build/              (akan berisi .pkl setelah training)
   ├── data/               (berisi dataset training)
   ├── src/
   │   ├── core_pipeline/
   │   └── model_training/
   ├── taxonomy.json       (skema sistem)
   └── context.md          (dokumentasi proyek)
   ```

---

### Langkah 1: Ekstraksi Fitur Data (Feature Engineering)

**Tujuan:** Mengubah dataset teks mentah menjadi representasi numerik yang siap untuk training ML.

**File:** `src/model_training/preprocess_features.py`

**Perintah:**
```powershell
cd d:\Project\Axolotl
python src/model_training/preprocess_features.py
```

**Proses Internal:**
1. Membaca file dataset (misal: `data/training_dataset.json` atau CSV dengan kolom: `[text, intent, BIO_tags]`)
2. Melakukan tokenization & normalisasi
3. Ekstrak fitur TF-IDF untuk SVM:
   - Menggunakan `TfidfVectorizer(ngram_range=(1,2))`
   - Menghasilkan matriks sparse dengan dimensi tetap
4. Ekstrak fitur kontekstual untuk CRF:
   - Untuk setiap token, ambil: `word`, `word.lower()`, `word[-3:]`, `word.isupper()`, `word.istitle()`, `POS_tag`, konteks ($w_{i-1}$, $w_i$, $w_{i+1}$)
5. Simpan output ke intermediate files (misal: `data/preprocessed/features_svm.pkl`, `data/preprocessed/features_crf.json`)

**Output Indikasi Sukses:**
```
[INFO] Loaded 1500 training samples
[INFO] Created SVM feature matrix: shape (1500, 2048)
[INFO] Created CRF feature sequences: 1500 sentences
[INFO] Preprocessing complete. Saved to data/preprocessed/
```

---

### Langkah 2: Pelatihan Model (Model Training)

**Tujuan:** Melatih SVM Intent Classifier dan CRF Slot Filler dari fitur yang sudah diekstrak.

**File:** `src/model_training/train_models.py`

**Perintah:**
```powershell
python src/model_training/train_models.py
```

**Proses Internal:**

#### 2a. SVM Intent Classification
```python
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score

# Load preprocessed features
X_train, y_train = load_svm_features()  # X: (1500, 2048), y: categorical [DIRECT, SCHEDULED, ...]

# Train SVM
svm_model = LinearSVC(
    C=1.0,
    loss='squared_hinge',
    penalty='l2',
    max_iter=1000,
    random_state=42
)
svm_model.fit(X_train, y_train)

# Cross-validation & metrics
cv_scores = cross_val_score(svm_model, X_train, y_train, cv=5)
print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Save
pickle.dump(svm_model, open("build/intent_model.pkl", "wb"))
pickle.dump(vectorizer, open("build/vectorizer.pkl", "wb"))
```

**Output Indikasi Sukses:**
```
[INFO] Training SVM...
[INFO] CV Accuracy: 0.9450 (+/- 0.0120)
[INFO] Saved intent_model.pkl
```

#### 2b. CRF Slot Filling
```python
import pycrfsuite

# Load preprocessed CRF features
X_train, y_train = load_crf_features()  # X: list of dicts, y: list of BIO tag sequences

# Train CRF
crf = pycrfsuite.Trainer()
for xseq, yseq in zip(X_train, y_train):
    crf.append(xseq, yseq)

crf.select('crf1d', 'lbfgs')
crf.set_params({
    'c1': 0.1,      # L1 coefficient
    'c2': 0.1,      # L2 coefficient
    'max_iterations': 100,
    'feature.possible_transitions': True
})

crf.train('build/slot_model.pkl')
```

**Output Indikasi Sukses:**
```
[INFO] Training CRF...
[INFO] CRF training complete. Saved slot_model.pkl
```

---

### Langkah 3: Validasi Model (Quality Assurance)

**Tujuan:** Memastikan model memiliki performa yang memuaskan sebelum deployment.

**File:** `src/model_training/verify_models.py`

**Perintah:**
```powershell
python src/model_training/verify_models.py
```

**Metrik yang Dievaluasi:**

1. **Intent Classification Metrics:**
   - Precision, Recall, F1-Score per class
   - Confusion Matrix
   - Overall Accuracy
   - Threshold: F1-Score ≥ 0.90 untuk semua class

2. **Slot Filling Metrics:**
   - Token-level accuracy
   - Sequence-level accuracy
   - F1-Score per slot type (ACTION, DIRECTION, QUANTITY, UNIT, TEMPORAL)
   - Threshold: Token accuracy ≥ 0.92

**Output Indikasi Sukses:**
```
========== INTENT CLASSIFICATION METRICS ==========
              precision    recall  f1-score
DIRECT         0.9500    0.9400    0.9450
SCHEDULED      0.9350    0.9450    0.9400
REPEATED       0.9100    0.9200    0.9150
STOP           0.9800    0.9700    0.9750
UNKNOWN        0.8900    0.9000    0.8950

Macro Avg:     0.9350    0.9350    0.9350

========== SLOT FILLING METRICS ==========
Token Accuracy: 0.9245
Sequence Accuracy: 0.8720
F1-Score per Slot:
  B-ACTION: 0.9520
  B-DIRECTION: 0.9380
  B-QUANTITY: 0.9100
  B-UNIT: 0.9450
  ...
```

**Jika Metrik Tidak Memenuhi Threshold:**
- Tambahkan lebih banyak training data
- Adjust hyperparameter (C, max_iterations)
- Kembali ke Langkah 1 (Feature Engineering)

---

### Langkah 4: Eksekusi Inferensi Hulu ke Hilir (Runtime Simulation)

**Tujuan:** Menjalankan seluruh pipa integrasi untuk menguji input teks hingga output JSON terstruktur.

**File:** `src/core_pipeline/e2e_simulation.py`

**Perintah:**

#### Mode Automated Testing (Default):
```powershell
python src/core_pipeline/e2e_simulation.py
```

#### Mode Interactive (Manual Testing):
```powershell
python src/core_pipeline/e2e_simulation.py --interactive
```

**Proses Internal:**

1. Load semua model dari `build/`:
   ```python
   pipeline = NLUPipeline(build_dir="build")
   # Loads: vectorizer.pkl, intent_model.pkl, slot_model.pkl
   ```

2. Execute 12 test cases automated:
   ```
   Test Case #1: "maju ke depan dua meter"
   Test Case #2: "mju dpan 2 mtr"
   ... [10 more test cases covering all intents, edge cases]
   ```

3. For each test case:
   - Normalize teks
   - Predict intent (SVM)
   - Predict slots (CRF)
   - Translate ke JSON via grounding engine
   - Print JSON payload + metadata

4. Simpan execution log ke: `src/core_pipeline/log/e2e_simulation.log`

**Output Indikasi Sukses:**
```
============================================================
       STARTING AUTOMATED PIPELINE INTEGRATION TESTS        
============================================================

[TEST CASE #1] Input: "maju ke depan dua meter"
{
  "status": "SUCCESS",
  "command": {
    "intent": "DIRECT_COMMAND",
    "action": "MOVE_FORWARD"
  },
  "parameters": {
    "spatial": {
      "direction": "FRONT",
      "quantity": 2.0,
      "unit": "METER"
    },
    "temporal": {
      "is_scheduled": false,
      "execute_at": null,
      "interval_quantity": null,
      "interval_unit": null
    }
  },
  "pipeline_metadata": {
    "fallback_triggered": false,
    "fallback_reason": null
  }
}

... [remaining test cases] ...

Automated integration tests completed successfully.
Full execution log successfully written to: D:\Project\Axolotl\src\core_pipeline\log\e2e_simulation.log
```

---

## 2. Bedah Isi Folder `build/` (Model Artifacts)

Folder `build/` menyimpan artefak hasil pelatihan dalam format biner ter-`pickle`. Setiap file adalah serialisasi Python object yang dapat di-load kembali dengan `pickle.load()`.

### Struktur & Fungsi Setiap File:

#### 1. **`vectorizer.pkl` (TF-IDF Vocabulary & Weights)**

**Tipe Object:** `sklearn.feature_extraction.text.TfidfVectorizer`

**Isi:**
- Vocabulary mapping: `{word → index}` untuk semua kata unik dari training data
- N-gram weights: Bobot TF-IDF untuk unigrams dan bigrams
- Configuration: `ngram_range=(1,2)`, `max_features=None`, `stop_words=None`

**Fungsi:**
- Mengubah kalimat baru (string) menjadi array vektor numerik berdimensi tetap
- Dimensi vektor = jumlah unique n-grams (biasanya 2000-3000)

**Cara Pakai:**
```python
import pickle
vectorizer = pickle.load(open("build/vectorizer.pkl", "rb"))
text = "maju ke depan 2 meter"
vector = vectorizer.transform([text])  # Returns sparse matrix (1, n_features)
print(vector.shape)  # Output: (1, 2048)
```

**Critical:** Jika kosakata training berbeda dengan inference, accuracy akan menurun. Gunakan vocabulary yang sama saat training dan inference!

---

#### 2. **`intent_model.pkl` (SVM Intent Classifier)**

**Tipe Object:** `sklearn.svm.LinearSVC`

**Isi:**
- Hyperplane weights untuk linear SVM
- Bias term
- Classes: `['DIRECT_COMMAND', 'SCHEDULED_COMMAND', 'REPEATED_COMMAND', 'STOP_COMMAND', 'UNKNOWN']`

**Fungsi:**
- Menerima vektor numerik dari `vectorizer` dan memprediksi class intent
- Output: class label (misal: `"DIRECT_COMMAND"`) + confidence score

**Cara Pakai:**
```python
import pickle
svm_model = pickle.load(open("build/intent_model.pkl", "rb"))
intent = svm_model.predict(vector)  # vector dari vectorizer
print(intent)  # Output: "DIRECT_COMMAND"
```

**Performance Constraints:**
- Inference time: <1ms per sample (sangat cepat)
- Memory footprint: ~5-10 MB

---

#### 3. **`slot_model.pkl` (CRF Slot Filler)**

**Tipe Object:** `pycrfsuite.Tagger` atau wrapper `CRFModel`

**Isi:**
- CRF state weights & transition probabilities
- Feature templates
- Labels: `['B-ACTION', 'I-ACTION', 'B-DIRECTION', 'I-DIRECTION', 'B-QUANTITY', 'I-QUANTITY', 'B-UNIT', 'I-UNIT', 'B-TEMPORAL', 'I-TEMPORAL', 'O']`

**Fungsi:**
- Menerima sequence of feature dicts (satu dict per token) dan memprediksi sequence of BIO tags
- Output: List of tags matching token length

**Cara Pakai:**
```python
crf_model = pickle.load(open("build/slot_model.pkl", "rb"))
token_features = [
    {"word": "maju", "pos": "VB", "word[-3:]": "aju", ...},
    {"word": "ke", "pos": "IN", "word[-3:]": "ke", ...},
    {"word": "depan", "pos": "NN", "word[-3:]": "pan", ...},
    ...
]
slot_tags = crf_model.predict(token_features)
print(slot_tags)  # Output: ['B-ACTION', 'O', 'B-DIRECTION', 'B-QUANTITY', 'B-UNIT']
```

**Performance Constraints:**
- Inference time: ~5-10ms per sequence (tergantung panjang)
- Memory footprint: ~20-30 MB

---

## 3. Maintenance & Updating Models

### Kapan Update Model?

1. **Accuracy Degradation:** Ketika deployment menghadapi input baru yang tidak diprediksi dengan baik.
   - Monitor: `fallback_triggered` di output JSON
   - Jika >10% queries trigger fallback, trigger retraining

2. **Data Distribution Shift:** Jika user mulai menggunakan pola linguistik baru yang tidak ada di training data.
   - Misal: Banyak user yang pakai abbreviasi baru yang tidak di-normalize dengan baik

3. **New Requirements:** Jika ada intent atau slot baru yang perlu ditambahkan.
   - Update `taxonomy.json` terlebih dahulu
   - Kumpulkan data baru untuk intent/slot
   - Rerun training pipeline

### Update Workflow:

1. **Collect New Data:** Kumpulkan queries dari production yang trigger fallback
2. **Annotate:** Label intent dan BIO tags untuk queries baru
3. **Merge:** Gabung dengan existing training data
4. **Retrain:** Jalankan ulang Langkah 1-4 dari section "Alur Eksekusi Lengkap"
5. **Validate:** Pastikan metrik tidak menurun untuk existing test cases
6. **Deploy:** Replace `.pkl` files di `build/` folder

---

## 4. Troubleshooting & Common Issues

### Issue #1: FileNotFoundError pada Model Loading
```
FileNotFoundError: Vectorizer not found at: build/vectorizer.pkl
```
**Solusi:** Pastikan training telah selesai dan `build/` folder berisi 3 file `.pkl`. Jika tidak, jalankan Langkah 1-2 dari "Alur Eksekusi Lengkap".

### Issue #2: Low Accuracy pada Test Cases
```
[RESULT] 7 dari 12 test cases SUCCESS, 5 FAILED/FALLBACK
```
**Solusi:** 
- Cek apakah training data representative terhadap test cases
- Tambah data, adjust hyperparameter, atau improve preprocessing
- Rerun training

### Issue #3: Import Error (`ModuleNotFoundError`)
```
ModuleNotFoundError: No module named 'sklearn'
```
**Solusi:**
```powershell
pip install scikit-learn python-crfsuite
```

---

## 5. Ringkasan Checklist Deployment

- [ ] Python 3.7+ terinstall
- [ ] Pustaka dependencies terinstall (`pip install -r requirements.txt`)
- [ ] Dataset ada di `data/`
- [ ] `taxonomy.json` sudah lock & consistent
- [ ] Langkah 1 (Feature Extraction) selesai
- [ ] Langkah 2 (Model Training) selesai
- [ ] Langkah 3 (Validation) metrik OK (F1 ≥ 0.90)
- [ ] Langkah 4 (E2E Simulation) semua test cases SUCCESS
- [ ] `build/` folder berisi 3 files: `vectorizer.pkl`, `intent_model.pkl`, `slot_model.pkl`
- [ ] Execution log ada di `src/core_pipeline/log/e2e_simulation.log`
- [ ] Ready untuk deployment ke ROS / Robot Controller
