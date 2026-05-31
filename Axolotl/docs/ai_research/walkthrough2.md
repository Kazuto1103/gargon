# Walkthrough - Phase 4: Pengembangan & Training Model ML

Fase 4 telah berhasil diselesaikan. Kami telah menginstal `python-crfsuite`, membangun skrip pelatihan dengan regularisasi ketat untuk mencegah overfitting, melatih model SVM dan CRF, mengevaluasi hasilnya secara detail, serta memverifikasi model pada uji prediksi kustom. Semua log eksekusi terminal disimpan dengan aman ke dalam folder `log/`.

## Perubahan yang Dilakukan

1. **Instalasi Pustaka Baru:**
   - Berhasil memasang `python-crfsuite` via pip untuk penanganan slot filling.

2. **Perubahan Peraturan Repositori (`rule.md`):**
   - Menambahkan aturan baru untuk menyimpan seluruh log eksekusi terminal (output run) dari skrip ke dalam folder [log/](file:///d:/Project/Axolotl/log).

3. **Pembuatan Skrip Pelatihan & Evaluasi:**
   - Menulis [train_models.py](file:///d:/Project/Axolotl/train_models.py).
   - Menggunakan `LinearSVC(C=0.5)` dengan 5-Fold Stratified Cross-Validation pada SVM Intent Classifier.
   - Mengimplementasikan `CRFModel` custom wrapper untuk mengemas `pycrfsuite` secara aman ke format `.pkl` dengan penambahan L1/L2 regularization (`c1=0.1`, `c2=0.1`).
   - Menyimpan seluruh cetakan terminal ke [log/train_models.log](file:///d:/Project/Axolotl/log/train_models.log) secara otomatis menggunakan kelas `DualWriter`.

4. **Pembuatan Skrip Verifikasi Model:**
   - Menulis [verify_models.py](file:///d:/Project/Axolotl/verify_models.py) untuk memuat model pickle dan melakukan prediksi teks bahasa Indonesia secara dinamis.
   - Hasil log disimpan ke [log/verify_models.log](file:///d:/Project/Axolotl/log/verify_models.log).

---

## Verifikasi dan Pengujian

### 1. Metrik Hasil Evaluasi Pelatihan Model (dikutip dari `log/train_models.log`)

* **Stratified 5-Fold CV Score (SVM):**
  - Rata-rata Akurasi: **99.36%** (std: 0.33%). Hal ini menunjukkan stabilitas spasial fitur yang sangat tinggi.
* **Evaluasi Test Set (SVM Intent):**
  - Akurasi rata-rata (macro/weighted average): **100% (F1-score: 1.00)** untuk semua kategori intent (`DIRECT`, `SCHEDULED`, `REPEATED`, `STOP`, `UNKNOWN`).
* **Evaluasi Test Set (CRF Slot, mengabaikan tag 'O'):**
  - F1-Score untuk seluruh label BIO (`B-ACTION`, `B-DIRECTION`, `B-QUANTITY`, dll): **1.00 (100% presisi)**.

### 2. Hasil Uji Prediksi Nyata (dikutip dari `log/verify_models.log`)

Berikut adalah beberapa hasil inferensi model pada kalimat uji baru:

* **Kalimat:** `"maju ke depan dua meter besok jam 3 sore"`
  - **Intent Terdeteksi:** `SCHEDULED_COMMAND`
  - **BIO Tagging:**
    - `maju` -> `B-ACTION`
    - `ke` -> `B-DIRECTION`
    - `depan` -> `I-DIRECTION`
    - `dua` -> `B-QUANTITY`
    - `meter` -> `B-UNIT`
    - `besok` -> `B-DATE`
    - `jam` -> `B-TIME`
    - `3` -> `I-TIME`
    - `sore` -> `I-TIME`

* **Kalimat:** `"batalkan semua gerakan sekarang juga"`
  - **Intent Terdeteksi:** `STOP_COMMAND`
  - **BIO Tagging:**
    - `batalkan` -> `B-ACTION`
    - `semua` -> `O`
    - `gerakan` -> `O`

* **Kalimat:** `"hari ini saya ingin memasak makan malam di rumah"` (Kasual/Noise)
  - **Intent Terdeteksi:** `UNKNOWN`
  - **BIO Tagging:** Seluruh kata diprediksi sebagai tag `O`.

---

## Model Terdaftar (Saved Binaries)
- [intent_model.pkl](file:///d:/Project/Axolotl/build/intent_model.pkl)
- [slot_model.pkl](file:///d:/Project/Axolotl/build/slot_model.pkl)
- [vectorizer.pkl](file:///d:/Project/Axolotl/build/vectorizer.pkl)
