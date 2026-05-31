Berikut adalah isi file `context.md` yang sangat detail dan terstruktur. File ini dirancang khusus untuk diletakkan di *root directory* proyek kamu agar AI Agent di IDE-mu memahami seluruh konteks, batasan, arsitektur, dan peta jalan (roadmap) proyek ini secara mendalam.

---

# PROJEK KONTEKS: GROUNDED NLP TEXT-TO-ACTION GENERATOR

## 1. Identitas Projek

* **Nama Projek:** Grounded NLP: Text-to-Action Sequence Generator Menggunakan Metode Intent-Slot Mapping untuk Otomasi Penjadwalan Perangkat Robotik.
* **Peran Pengguna (User):** Lead Project / Product Owner.
* **Peran AI Agent:** Senior NLP Architect, ML Engineer, & Robotics Ontologist.
* **Fokus Bahasa:** Bahasa Indonesia (Formal, Semi-formal, dan Percakapan Sehari-hari).
* **Fokus Ranah (Domain):** Kontrol Gerakan Robot Universal (Beroda, Berkaki, atau Lengan Sendi) dengan Lapisan Penjadwalan Waktu (Scheduling Layer).

---

## 2. Deskripsi Projek & Arsitektur Sistem

Projek ini bertujuan untuk membangun sistem *Natural Language Understanding* (NLU) lokal (*closed-domain*) yang menerima input berupa perintah ketikan teks dalam Bahasa Indonesia, mengekstrak makna serta parameternya, lalu menerjemahkannya menjadi urutan aksi terstruktur (JSON) yang siap dieksekusi atau dijadwalkan pada perangkat robotik universal.

Sistem ini memisahkan logika pemahaman bahasa dari perangkat keras robot melalui abstraksi gerakan dasar (maju, mundur, belok, dll), besaran (jarak/sudut), modifier (kecepatan), serta waktu eksekusi.

### Arsitektur Teknologi (ML Stack)

* **Intent Classification (Klasifikasi Niat):** Menggunakan algoritma **SVM (Support Vector Machine)** untuk menentukan tipe perintah (Langsung, Terjadwal, Berulang, atau Berhenti).
* **Slot Filling (Pengisian Parameter):** Menggunakan algoritma **CRF (Conditional Random Fields)** dengan format **BIO Tagging** untuk mengekstrak entitas kata (Action, Direction, Quantity, Unit, Modifier, Time, Date).
* **Action Sequence Generator (Grounding Modul):** Aturan berbasis kode (Rule Engine) untuk memetakan token bahasa menjadi konstanta robotik universal dan memparsing teks waktu relatif menjadi *absolute timestamp ISO 8601*.

---

## 3. Spesifikasi Output Sistem

Sistem wajib menghasilkan output akhir berupa struktur data JSON standar yang siap dikonsumsi oleh *scheduler* atau *microcontroller* robot.

### Contoh Kontrak Output JSON:

```json
{
  "intent": "SCHEDULED_COMMAND",
  "execution_type": "SCHEDULED",
  "schedule_timestamp": "2026-05-31T12:00:00Z",
  "sequence": {
    "action": "MOVE",
    "direction": "LEFT",
    "magnitude": 2.0,
    "unit": "METER",
    "speed_modifier": 0.5
  }
}

```

---

## 4. Fase-Fase Pengerjaan Projek

Projek ini dikerjakan secara bertahap melalui 6 fase linear berikut:

### Fase 1: Desain Taksonomi & Konfigurasi (`taxonomy.json`) `[FASE SEKARANG]`

* **Target:** Mengunci skema Intent, Slot, Batasan Kamus Kosakata (Vocabulary Boundary), dan Mapping Grounding awal.
* **Output:** File `taxonomy.json` yang menjadi acuan tunggal (*Single Source of Truth*) sistem.

### Fase 2: Pembuatan Dataset Sintetis & Pelabelan

* **Target:** Membuat skrip otomatisasi (*data augmentation*) menggunakan Python untuk menghasilkan ribuan variasi kalimat berdasarkan kamus di Fase 1.
* **Output:** Dataset latih (`dataset.csv` atau `dataset.json`) yang sudah terlabeli dengan format Intent dan BIO Tagging untuk CRF.

### Fase 3: Pra-pemrosesan Data & Ekstraksi Fitur

* **Target:** Membangun *pipeline* pengolahan teks Bahasa Indonesia (Case folding, tokenization) dan rekayasa fitur linguistik.
* **Output:** Ekstraktor fitur berbasis TF-IDF/Word Embedding (untuk SVM) dan fitur kontekstual/POS-Tagging per kata (untuk CRF).

### Fase 4: Pengembangan & Training Model ML

* **Target:** Melatih model SVM dan CRF menggunakan pustaka seperti Scikit-Learn dan FastCRF/python-crfsuite.
* **Output:** File model biner (`.pkl` atau `.bin`) yang siap pakai beserta laporan evaluasi akurasi (Precision, Recall, F1-Score).

### Fase 5: Pembangunan Modul Grounding (Sequence Generator)

* **Target:** Membuat komponen logika yang menerima prediksi dari Fase 4, melakukan validasi batasan parameter, dan mentranslasikan teks waktu menjadi format waktu sistem.
* **Output:** Modul Python/Script Translator yang menghasilkan output JSON standar.

### Fase 6: Pengujian End-to-End & Simulasi

* **Target:** Menguji ketahanan model terhadap kalimat baru, typo ringan, dan struktur kalimat tidak lengkap (Penanganan *Missing Slots*).
* **Output:** Sistem aplikasi NLU utuh (*ready-to-deploy*) yang lolos uji skenario integrasi.

---

## 5. Batasan & Prinsip Kerja AI Agent

Ketika membantu Lead Project dalam menulis kode atau menyusun data, AI Agent harus mematuhi aturan berikut:

1. **Strictly Closed-Domain:** Jangan berasumsi sistem ini terhubung ke LLM eksternal atau internet. Semua ekstraksi bergantung pada pola yang dilatih melalui SVM + CRF.
2. **Indonesian-Centric:** Utamakan fleksibilitas tata bahasa Indonesia (misal: penulisan angka "dua" vs "2", penulisan waktu "jam 12" vs "pukul 12", kata penunjuk "nanti", "besok").
3. **Deterministic Output:** Modul Grounding harus mengubah kata sinonim menjadi satu token instruksi yang seragam (Contoh: "kiri", "mengiri", "ke kiri" semuanya harus diterjemahkan menjadi `"LEFT"`).
4. **No Placeholders:** Setiap kode, dataset, atau skema konfigurasi yang diminta oleh Lead Project harus ditulis lengkap tanpa pemotongan (`...`).