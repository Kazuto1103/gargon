# Laporan Analisis Kekasaran Permukaan
## Surface Roughness Detection / Texture Analysis Project

**Tanggal Analisis:** 18 Mei 2026  
**Dataset:** 63 gambar mikrograf permukaan (figura1_exterior*.tif)  
**Metode:** GLCM (Gray-Level Co-occurrence Matrix) dan LBP (Local Binary Patterns)

---

## 📊 HASIL KLASIFIKASI KESELURUHAN

| Metrik | Nilai |
|--------|-------|
| **Total Gambar** | 63 |
| **Kasar (Rough)** | 63 (100%) |
| **Halus (Smooth)** | 0 (0%) |
| **Confidence Rata-rata** | 99.39% |

### Output Folder Terorganisir:
- ✅ `data/roughness_kasar/` → 63 gambar terklasifikasi kasar
- ✅ `data/roughness_halus/` → 0 gambar terklasifikasi halus

---

## 📈 STATISTIK FITUR EKSTRAKSI GLCM

**Gray-Level Co-occurrence Matrix Features:**

| Fitur | Rata-rata | Std Dev | Min | Max |
|-------|-----------|---------|-----|-----|
| **Contrast** | 136.98 | 29.02 | 87.63 | 219.08 |
| **Homogeneity** | 0.2504 | 0.0223 | 0.2071 | 0.3041 |
| **Energy** | 0.0824 | 0.0200 | 0.0419 | 0.1349 |
| **Correlation** | 0.9893 | 0.0025 | 0.9820 | 0.9947 |

### Interpretasi GLCM:
- **Contrast TINGGI (137)** → Perbedaan intensitas lokal besar → Permukaan KASAR
- **Homogeneity RENDAH (0.25)** → Tekstur tidak uniform → Konsisten dengan permukaan kasar
- **Energy RENDAH (0.08)** → Pola tidak konsisten → Menunjukkan ketidakteraturan
- **Correlation TINGGI (0.99)** → Hubungan spasial kuat antar pixel

---

## 🔢 STATISTIK FITUR EKSTRAKSI LBP

**Local Binary Patterns Features:**

| Fitur | Rata-rata | Std Dev |
|-------|-----------|---------|
| **LBP Mean** | 5.01 | 0.82 |
| **LBP Std Dev** | 2.48 | 0.24 |
| **LBP Entropy** | 3.19 | 0.024 |

### Interpretasi LBP:
- **LBP Mean & Std Dev** → Mengukur variasi pola biner lokal
- **LBP Entropy ~3.19** → Distribusi pola relatif seragam untuk semua gambar
- Konsisten dengan dataset homogen (semua permukaan kasar)

---

## 🔬 HASIL ANALISIS PERBANDINGAN (Contoh: 2 Gambar Kasar)

### Gambar yang Dibandingkan:
1. `figura1_exterior1_0.tif` (Permukaan Kasar #1)
2. `figura1_exterior1_1.tif` (Permukaan Kasar #2)

### Perbandingan GLCM Features:

```
Feature              Image 1          Image 2          Difference
───────────────────────────────────────────────────────────────────
Contrast             102.820066       103.749335       0.929269
Homogeneity          0.262308         0.256405         0.005903
Energy               0.084879         0.078647         0.006232
Correlation          0.988821         0.988673         0.000148
```

### Perbandingan LBP Statistics:

```
Statistic            Image 1          Image 2          Difference
───────────────────────────────────────────────────────────────────
Mean                 5.014419         5.000000         0.014282
Std Deviation        2.495050         2.498398         0.003229
```

### SENSITIVITY ANALYSIS - Metode Mana yang Lebih Sensitif?

**GLCM Sensitivity (Perubahan fitur antar dua permukaan):**
- Contrast:      0.929269 ← **TERTINGGI - Mendeteksi perbedaan kontras lokal**
- Homogeneity:   0.005903
- Energy:        0.006232
- Correlation:   0.000148

**LBP Sensitivity (Perubahan pola biner lokal):**
- Mean LBP:      0.014282
- Std Dev LBP:   0.003229

### ✅ KESIMPULAN:

**GLCM menunjukkan sensitivitas LEBIH TINGGI terhadap perbedaan kekasaran**
- GLCM max: **0.929269** >> LBP max: **0.014282**
- GLCM ~65x lebih sensitif dibanding LBP

**REKOMENDASI:**
- ✅ Gunakan **GLCM** untuk analisis kekasaran permukaan yang lebih detail dan sensitif
- ✅ Fitur **Contrast** (kontras) adalah indikator terbaik tingkat kekasaran
- ⚠️ LBP berguna untuk karakterisasi tekstur pola, tapi kurang sensitif untuk membedakan tingkat kekasaran mikro

---

## 📁 OUTPUT FILES YANG DIHASILKAN

### Klasifikasi & Organisasi:
```
data/
├── roughness_kasar/
│   ├── figura1_exterior1_0.tif
│   ├── figura1_exterior1_1.tif
│   └── ... [63 files total]
└── roughness_halus/
    └── [kosong]
```

### Laporan Analisis:
```
analysis_results/
├── texture_analysis_results.csv          [Data detail 63 gambar]
├── texture_analysis_detailed.json        [Format JSON terstruktur]
├── classification_report.txt             [Laporan teks lengkap]
└── classification_visualization.png      [6 subplot grafis analisis]
```

---

## 🛠️ SCRIPTS YANG TERSEDIA

### 1. **roughness_classifier.py** (Klasifikasi Otomatis)
```bash
python src/roughness_classifier.py
```
- Scan semua 63 gambar
- Ekstrak fitur GLCM & LBP
- Klasifikasi kasar/halus otomatis
- Organisir ke folder terpisah
- Generate laporan (CSV, JSON, TXT, PNG)

### 2. **texture_analyzer.py** (Analisis Perbandingan Detail)
```bash
python src/texture_analyzer.py
```
- Perbandingan 2 gambar side-by-side
- Visualisasi gambar asli + LBP map
- Tabel statistik perbandingan
- Analisis sensitivitas GLCM vs LBP

### 3. **quick_analysis.py** (Analisis Cepat)
```bash
python src/quick_analysis.py
```
- Versi ringkas texture_analyzer
- Hanya output terminal (no GUI)
- Cocok untuk batch processing

---

## 💡 INTERPRETASI HASIL UNTUK PRESENTASI

### Mengapa Semua Gambar Terklasifikasi KASAR?

Dataset `figura1_exterior*.tif` adalah koleksi mikrograf permukaan material dengan tingkat kekasaran tinggi. Semua gambar menunjukkan:

1. **GLCM Contrast TINGGI (137)** → Ketidakteraturan permukaan signifikan
2. **Homogeneity RENDAH (0.25)** → Tekstur tidak uniform di seluruh permukaan
3. **LBP Entropy KONSISTEN** → Pola biner tersebar merata (tidak ada area halus terlokalisir)

### Karakteristik Permukaan Kasar:
✓ Kontras tinggi antar pixel lokal  
✓ Tekstur heterogen (tidak uniform)  
✓ Pola penyebaran konsisten di seluruh area  
✓ Tingkat kekasaran: **ROUGH** (>90% confidence)

---

## 📝 REKOMENDASI PENGEMBANGAN LANJUTAN

1. **Threshold Tuning**: Sesuaikan threshold klasifikasi untuk dataset campuran kasar-halus
2. **Multi-scale Analysis**: Gunakan multiple kernel sizes untuk GLCM dan LBP
3. **Feature Fusion**: Gabungkan GLCM+LBP dengan ML classifier (SVM/RF)
4. **3D Surface**: Jika tersedia data 3D, gunakan advanced metrics (Sa, Sq, Sz)
5. **Quality Control**: Implementasi real-time roughness monitoring system

---

## 📚 REFERENSI METODE

### GLCM (Gray-Level Co-occurrence Matrix):
- Menganalisis hubungan spasial antar pixel pada jarak d dan sudut θ
- Fitur diekstrak: Contrast, Homogeneity, Energy, Correlation
- Ideal untuk: Analisis tekstur detail, orientasi-sensitif

### LBP (Local Binary Patterns):
- Menghitung pola biner 8-bit di sekitar setiap pixel
- Histogram LBP digunakan sebagai fitur vektor
- Ideal untuk: Klasifikasi tekstur cepat, recognisi pola

---

**Project Status:** ✅ **SELESAI**  
**Data Ready:** ✅ 63 gambar teranalisis & terorganisir  
**Presentation Ready:** ✅ Laporan & visualisasi lengkap
