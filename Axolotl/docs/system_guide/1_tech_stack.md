# Dokumentasi Sistem NLU - Teknologi & Arsitektur Sistem

Dokumen ini ditujukan bagi Software Engineer / AI Engineer baru untuk memahami landasan teknologi, paradigma desain hibrida, dan ekosistem teknologi di balik modul Natural Language Understanding (NLU) universal robot ini.

**Konteks Proyek:** Sistem ini dirancang untuk memahami perintah teks dalam Bahasa Indonesia (formal, semi-formal, dan sehari-hari) dan menerjemahkannya menjadi urutan aksi terstruktur (JSON) yang siap dieksekusi pada perangkat robotik universal (beroda, berkaki, atau lengan sendi) dengan dukungan penjadwalan waktu.

---

## 1. Paradigma Sistem: Arsitektur Hibrida (Hybrid Architecture)

Sistem ini tidak bergantung 100% pada Machine Learning (ML). Kami menerapkan pendekatan **Hibrida Tiga Lapis**:

### Lapisan 1: Preprocessing & Normalisasi (Deterministic)
* **Fungsi:** Membersihkan input teks, memperbaiki penulisan singkatan/gaul Bahasa Indonesia, menghilangkan kata noise, dan memisahkan angka dari satuan yang menempel.
* **Keuntungan:** Mengurangi variasi input sebelum masuk ke model ML, meningkatkan akurasi downstream.
* **Contoh:** `"mju dpan 2 mtr"` → `"maju ke depan 2 meter"`

### Lapisan 2: Prediksi ML (Stochastic)
* **Intent Classification (SVM):** Menentukan tipe perintah: `DIRECT_COMMAND`, `SCHEDULED_COMMAND`, `REPEATED_COMMAND`, atau `STOP_COMMAND`.
* **Slot Filling (CRF):** Mengekstrak parameter seperti ACTION, DIRECTION, QUANTITY, UNIT, TEMPORAL_MODIFIER, dsb.
* **Fleksibilitas:** Menangani variasi bahasa manusia yang acak dan tidak terprediksi sebelumnya.

### Lapisan 3: Grounding & Validasi (Deterministic + Safety-Critical)
* **Fungsi:** Filter keamanan akhir yang menerjemahkan prediksi ML menjadi instruksi robot tervalidasi.
* **Keamanan:** Validasi batasan parameter, deteksi anomali, fallback ke nilai default aman jika terjadi kesalahan.
* **Determinism:** Sinonim bahasa → satu token instruksi yang seragam (misal: `"kiri"`, `"mengiri"`, `"ke kiri"` semua menjadi `LEFT`).

**Filsafat Desain:** Lapisan 1 & 3 mencegah *garbage in, garbage out*, sementara Lapisan 2 memberikan fleksibilitas untuk menangani variasi linguistik.

---

## 2. Komponen Komputasi & Pustaka Utama

Proyek ini dibangun menggunakan **Python 3.x** dengan pustaka-pustaka kelas industri berikut:

### A. Scikit-Learn (`scikit-learn`)
* **Fungsi:** Pemrosesan teks berbasis statistik dan klasifikasi intensi (*Intent Classification*).
* **Komponen Teknis:**
  - `TfidfVectorizer`: Mengubah teks menjadi matriks fitur numerik menggunakan Term Frequency-Inverse Document Frequency dengan N-gram (1,2).
  - `LinearSVC`: Classifier Support Vector Machine dengan kernel linear untuk klasifikasi multi-kelas yang cepat dan memory-efficient.
* **Alasan Pemilihan:** 
  - SVM cepat, efisien dalam penggunaan memori, dan sangat robust untuk klasifikasi teks multi-kelas pada dataset kecil-menengah.
  - TF-IDF menangkap pentingnya kata dalam konteks tanpa perlu embedding yang kompleks.
  - Output: Prediksi intent dengan confidence score.

### B. Python-CRFsuite (`python-crfsuite`)
* **Fungsi:** Pengenalan Entitas Bernama / Ekstraksi Slot Informasi (*Named Entity Recognition / Sequence Labeling*).
* **Komponen Teknis:**
  - Format Label: BIO Tagging (`B-ACTION`, `I-ACTION`, `B-DIRECTION`, `I-DIRECTION`, `B-QUANTITY`, `I-QUANTITY`, `B-UNIT`, `I-UNIT`, `O`).
  - Regularisasi: L1/L2 dengan koefisien ($c_1=0.1, c_2=0.1$) untuk mencegah overfitting.
  - Fitur Kontekstual: Mempertimbangkan token sebelumnya ($w_{i-1}$), token saat ini ($w_i$), dan token berikutnya ($w_{i+1}$).
* **Alasan Pemilihan:**
  - CRF mempertimbangkan konteks sekuen token sehingga sangat akurat dalam membaca struktur kalimat perintah robotik.
  - Dapat menangani urutan label yang kompleks dan interdependen.
  - Output: Sequence tag per token beserta confidence.

### C. Regex & Standard Libraries (`re`, `json`, `pickle`)
* `re`: Engine pembersihan singkatan dan normalisasi teks Bahasa Indonesia.
* `json`: Serialisasi output standar JSON untuk komunikasi dengan subsistem lain (scheduler, ROS).
* `pickle`: Serialisasi model biner untuk deployment dan inference cepat.

### D. Taxonomy Reference (`taxonomy.json`)
* **Fungsi:** Single source of truth untuk skema Intent, Slot, kamus kosakata, dan mapping grounding.
* **Isi:**
  - Intent categories dengan deskripsi.
  - Slot definitions dengan tipe data.
  - Vocabulary boundaries (action verbs, direction keywords, temporal units).
  - Grounding mappings (bahasa → token robotik).
* **Penggunaan:** Di-load saat inisialisasi pipeline untuk validasi dan mapping.

---

## 3. Alur Aliran Data (Hulu ke Hilir)

```
Teks Input (Raw String)
   ↓
[normalizer.py] 
   Fungsi: Penghapusan kata noise & standardisasi singkatan
   Input:  "mju dpan 2 mtr"
   Output: "maju ke depan 2 meter"
   ↓
[preprocess_features.py]
   Fungsi: Ekstraktor fitur TF-IDF dan kontekstual
   Output: Vektor numerik untuk SVM, fitur BIO untuk CRF
   ↓
[SVM Model] (Intent Classification)
   Fungsi: Deteksi kategori intent
   Input:  Vektor TF-IDF
   Output: Intent + confidence (misal: "DIRECT_COMMAND", 0.95)
   ↓
[CRF Model] (Slot Filling)
   Fungsi: Ekstraksi BIO-Tags tiap Kata
   Input:  Fitur kontekstual per token
   Output: [(token, tag), ...] misal: [("maju", "B-ACTION"), ("ke", "O"), ("depan", "B-DIRECTION"), ("2", "B-QUANTITY"), ("meter", "B-UNIT")]
   ↓
[grounding_translator.py] (Grounding Engine)
   Fungsi: Resolusi konflik, validasi tipe, pengikatan slot ke taksonomi
   Input:  Intent + token-tag pairs
   Output: Standardized JSON Command Payload
   ↓
Output Akhir (Structured JSON)
   Siap dikonsumsi ROS / Scheduler / Perangkat Keras Robot
```

---

## 4. Kontrak Output JSON Standar

Setiap prediksi harus menghasilkan struktur JSON yang konsisten:

```json
{
  "status": "SUCCESS",
  "command": {
    "intent": "SCHEDULED_COMMAND",
    "action": "MOVE_FORWARD"
  },
  "parameters": {
    "spatial": {
      "direction": "LEFT",
      "quantity": 2.0,
      "unit": "METER"
    },
    "temporal": {
      "is_scheduled": true,
      "execute_at": "2026-06-01T09:00:00Z",
      "interval_quantity": null,
      "interval_unit": null
    }
  },
  "pipeline_metadata": {
    "fallback_triggered": false,
    "fallback_reason": null
  }
}
```

**Catatan Penting:**
- `status`: `SUCCESS` jika intent valid, `REJECTED` jika intent adalah `UNKNOWN`.
- `fallback_triggered`: Menandakan apakah sistem menggunakan nilai default karena ekstraksi slot gagal.
- Semua nilai numerik dalam format float, satuan dalam huruf besar, arah dalam format enum standar (FRONT, BACK, LEFT, RIGHT, UP, DOWN).

---

## 5. Indonesian Language-Specific Considerations

Sistem dirancang khusus untuk fleksibilitas Bahasa Indonesia:

### Variasi Numerik
- "dua meter" vs "2 meter" vs "2m" vs "2meter" ← semua harus terdeteksi sebagai quantity=2, unit=METER

### Variasi Penulisan Waktu
- "besok jam 3 sore" vs "nanti pukul 3 PM" vs "esok tengah siang" ← parser relative time harus menangani semua variasi

### Sinonim & Gaul
- "maju" ≈ "gerak maju" ≈ "jalan ke depan" ← SVM + CRF harus belajar pola-pola sinonim dari training data

### Kata Noise & Filler
- "Woi robot, maju dong!" ← normalizer harus menghilangkan "woi" dan "dong" sebelum processing

---

## 6. Fase Pengembangan & Integrasi

Sistem NLU ini adalah bagian dari Fase 6 dari roadmap proyek:

1. **Fase 1-4:** Persiapan data, feature engineering, training model (sudah selesai).
2. **Fase 5:** Grounding layer & Sequence Generator (sudah selesai).
3. **Fase 6 (Current):** E2E Testing & Simulation ← Dokumentasi ini berlaku di sini.
4. **Future:** Integrasi dengan ROS/Scheduler, deployment ke hardware robot.
