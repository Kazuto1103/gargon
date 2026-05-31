# Walkthrough - Patch Integrasi Normalizer Pipeline NLU

Sesi ini berfokus pada penguatan ketahanan model terhadap teks kotor bahasa Indonesia dengan cara memisahkan logika normalisasi ke dalam modul khusus dan mengintegrasikannya secara konsisten ke seluruh pipeline.

## Perubahan yang Dilakukan

### 1. Modul Baru: [normalizer.py](file:///d:/Project/Axolotl/normalizer.py)
- Membuat `NORM_DICT` berisi 40+ entri pemetaan kosa kata gaul/singkatan bahasa Indonesia ke kata-kata standar dalam taksonomi.
- Fungsi `normalize_text(text: str) -> str`: menangani 3 tahap normalisasi:
  1. **Spasi angka-satuan**: `"10cm"` → `"10 centimeter"`, `"90deg"` → `"90 derajat"`
  2. **Substitusi token dari NORM_DICT**: `"mju"` → `"maju"`, `"kekiri"` → `"ke kiri"`, `"ntar"` → `"nanti"`, dll.
  3. **Pembersihan kata noise**: token seperti `"heii"`, `"woi"`, `"dong"`, `"mang"` dihapus.
- Fungsi `tokenize_sentence(text: str) -> list[str]`: menjadi **satu-satunya tokenizer kanonik** di seluruh pipeline.

### 2. Patch: [preprocess_features.py](file:///d:/Project/Axolotl/preprocess_features.py)
- Import `normalize_text`, `tokenize_sentence` dari `normalizer.py`.
- `preprocess_text()` sekarang menerapkan `normalize_text()` sebelum pembersihan tanda baca. Ini memastikan TF-IDF SVM dilatih dari kosa kata yang sudah baku.
- Fitur CRF tetap menggunakan token asli dataset sintetis (untuk menjaga keselarasan BIO-tag); normalisasi diterapkan hanya pada inferensi kalimat baru dari pengguna.

### 3. Patch: [stress_test_models.py](file:///d:/Project/Axolotl/stress_test_models.py)
- Import terpusat dari `normalizer.py`, menghapus duplikasi `tokenize_sentence` lokal.
- Setiap kalimat uji dinormalisasi terlebih dahulu sebelum vektorisasi SVM dan ekstraksi fitur CRF.
- Log output kini menampilkan **[Normalized]** untuk setiap kasus uji.

### 4. Patch: [verify_models.py](file:///d:/Project/Axolotl/verify_models.py)
- Pola yang sama: import dari `normalizer.py`, normalisasi sebelum inferensi.

---

## Perbandingan Hasil Stress Test (Before vs After)

| Kasus Uji | Intent (Sebelum) | Intent (Sesudah) | Perbaikan Kunci |
|-----------|-----------------|-----------------|-----------------|
| `"mju dpan 2 mtr ntar jm 3 sor"` | `DIRECT_COMMAND` ❌ | `DIRECT_COMMAND` (slot waktu kini lebih baik) | `mju→maju`, `mtr→meter`, `ntar→nanti` |
| `"geser kekiri dua meter bsk pagi"` | `SCHEDULED_COMMAND` ✅ | `SCHEDULED_COMMAND` ✅ | `kekiri→ke kiri` diparsing sebagai `B-DIRECTION I-DIRECTION` |
| `"heii tolong gerak kekanan dikit dong"` | Slot: `kekanan → B-DIRECTION` (tanpa I-) | Slot: `ke→B-DIRECTION, kanan→I-DIRECTION` ✅ | `heii`, `dong` dihapus, `kekanan→ke kanan` |
| `"mundur 10cm skrg"` | Slot: `10cm→B-QUANTITY` (satuan tergabung) | Slot: `10→B-QUANTITY, centimeter→B-UNIT` ✅ | `10cm→10 centimeter` via regex spasi angka-satuan |
| `"tiap 5 mnt belok kekanan 90 deg"` | `mnt→B-UNIT`, `deg` tidak dikenali | `menit→B-UNIT`, `derajat→B-UNIT` ✅ | `mnt→menit`, `deg→derajat`, `kekanan→ke kanan` |
| `"woi robot berhenti secepatnya dong"` | Intent: `STOP`, slot `berhenti→O` | Intent: `STOP`, `woi`/`dong` bersih | Kata noise dieliminasi dari kalimat |

---

## Catatan Arsitektural Penting

> [!NOTE]
> **Mengapa CRF training tidak menggunakan normalisasi?** Normalisasi dapat mengubah jumlah token (misal: `"kekiri"` → `"ke kiri"` dari 1 token menjadi 2 token), yang akan merusak keselarasan satu-ke-satu antara token dan BIO-tag pada dataset sintetis. Oleh karena itu:
> - **Training CRF**: Menggunakan token asli dari dataset (sudah bersih dan selaras).
> - **Inferensi CRF pada input baru**: Menerapkan normalisasi penuh sebelum tokenisasi.

## File Log Terbaru
- [log/preprocess_features.log](file:///d:/Project/Axolotl/log/preprocess_features.log)
- [log/train_models.log](file:///d:/Project/Axolotl/log/train_models.log)
- [log/stress_test.log](file:///d:/Project/Axolotl/log/stress_test.log)
- [log/verify_models.log](file:///d:/Project/Axolotl/log/verify_models.log)
