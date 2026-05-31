# Walkthrough - Phase 3: Pra-pemrosesan Data & Ekstraksi Fitur

Fase 3 telah diselesaikan dengan sukses. Kami berhasil merancang, menulis, mengeksekusi pipeline pra-pemrosesan data, melakukan ekstraksi fitur untuk SVM dan CRF, membagi data (80% train / 20% test), serta menverifikasi kebenaran serialisasi.

## Perubahan yang Dilakukan

1. **Pembuatan Skrip Pra-pemrosesan & Ekstraksi Fitur:**
   - Menulis [preprocess_features.py](file:///d:/Project/Axolotl/preprocess_features.py) dengan pipeline feature engineering lengkap.
   - Menggunakan `TfidfVectorizer` scikit-learn dengan parameter `ngram_range=(1, 2)` untuk mengekstrak n-gram kata pada SVM.
   - Mendesain ekstraktor fitur linguistik tingkat kata untuk CRF dengan jendela kontekstual ($i-1$, $i$, $i+1$) lengkap dengan deteksi akhiran bahasa Indonesia (`-kan`, `-an`, `-i`), angka, dan penanda batas kalimat (`BOS`, `EOS`).
   - Melakukan serialisasi bertipe biner menggunakan `pickle` ke folder terpusat `build/`.

2. **Pembuatan Skrip Verifikasi Fitur:**
   - Menulis [verify_preprocessed.py](file:///d:/Project/Axolotl/verify_preprocessed.py) untuk memastikan keberadaan berkas biner, rasio pembagian data, kecocokan dimensi matriks TF-IDF, serta konsistensi jumlah token dan slot pada fitur CRF.

---

## Verifikasi dan Pengujian

Kami mengeksekusi ekstraksi fitur dan menjalankan pengujian integritas:

### 1. Eksekusi Ekstraksi Fitur:
```powershell
python preprocess_features.py
```
**Output:**
```text
Loading dataset from: D:\Project\Axolotl\synthetic_dataset.json
Splitting dataset into 80% train and 20% test sets (stratified by intent)...
Building SVM features...
Building CRF features...
Serializing vectorizer and datasets...
Success: All components have been serialized to the 'build/' directory!
 - Vectorizer: D:\Project\Axolotl\build\vectorizer.pkl
 - SVM Data: D:\Project\Axolotl\build\svm_data.pkl
 - CRF Data: D:\Project\Axolotl\build\crf_data.pkl
```

### 2. Jalankan Pengujian Validitas File Aset:
```powershell
python verify_preprocessed.py
```
**Output:**
```text
Starting verification of build files in: D:\Project\Axolotl\build
Verifying vectorizer.pkl...
 - Vectorizer fitted. Vocabulary size: 2910
Verifying svm_data.pkl...
 - SVM Train size: 2200
 - SVM Test size: 550
 - SVM X_train matrix shape: (2200, 2910)
 - SVM X_test matrix shape: (550, 2910)
Verifying crf_data.pkl...
 - CRF Train size: 2200
 - CRF Test size: 550

SUCCESS: All serialized features and splits are completely INTEGRAL and VALID!
```

> [!TIP]
> Fitur n-gram 1-2 menghasilkan dimensi kosa kata unik sebanyak 2910 dimensi, yang memberikan representasi fitur spasial dan temporal yang sangat baik untuk model intent classifier.
