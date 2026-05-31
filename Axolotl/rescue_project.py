import os
import shutil

# 1. Definisikan Struktur Folder
FOLDERS = [
    "build",
    "data",
    "docs/ai_research",
    "docs/system_guide",
    "logs",
    "src/model_training",
    "src/core_pipeline"
]

# 2. Definisikan Pemetaan File (Asal -> Tujuan)
FILE_MAPPING = {
    # Kor biner model
    "vectorizer.pkl": "build/vectorizer.pkl",
    "intent_model.pkl": "build/intent_model.pkl",
    "slot_model.pkl": "build/slot_model.pkl",
    # Dataset
    "synthetic_dataset.json": "data/synthetic_dataset.json",
    # Dokumen riset lama
    "walkthrough.md": "docs/ai_research/walkthrough.md",
    "walkthrough2.md": "docs/ai_research/walkthrough2.md",
    "walkthrough3.md": "docs/ai_research/walkthrough3.md",
    # Kode Kelompok A (Training)
    "preprocess_features.py": "src/model_training/preprocess_features.py",
    "train_models.py": "src/model_training/train_models.py",
    "verify_preprocessed.py": "src/model_training/verify_preprocessed.py",
    "verify_models.py": "src/model_training/verify_models.py",
    # Kode Kelompok B (Runtime Engine)
    "normalizer.py": "src/core_pipeline/normalizer.py",
    "grounding_translator.py": "src/core_pipeline/grounding_translator.py",
    "e2e_simulation.py": "src/core_pipeline/e2e_simulation.py",
}

# 3. Konten Dokumentasi Markdown
DOCS = {
    "docs/system_guide/1_tech_stack.md": """# Dokumentasi Sistem NLU - Teknologi & Arsitektur Sistem

Dokumen ini ditujukan bagi Software Engineer / AI Engineer baru untuk memahami landasan teknologi dan paradigma berpikir di balik modul Natural Language Understanding (NLU) universal robot ini.

## 1. Paradigma Sistem: Arsitektur Hibrida (Hybrid Architecture)
Sistem ini tidak bergantung 100% pada Machine Learning (ML). Kami menerapkan pendekatan **Hibrida**:
* **Lapisan Prediktif (ML):** Digunakan untuk menangani variasi bahasa manusia yang acak (Klasifikasi Intent dan Slot Filling).
* **Lapisan Deterministik (Rule-Based Grounding):** Digunakan sebagai filter akhir (Interpreter) guna memastikan output yang dikirim ke perangkat keras robot 100% aman, tervalidasi, dan bebas dari halusinasi model AI.

## 2. Komponen Komputasi & Pustaka Utama
Projek ini dibangun menggunakan **Python 3.x** dengan pustaka-pustaka kelas industri berikut:

### A. Scikit-Learn (`scikit-learn`)
* **Fungsi:** Pemrosesan teks berbasis statistik dan klasifikasi intensi (*Intent Classification*).
* **Algoritma:** `LinearSVC` (Support Vector Machine) dikombinasikan dengan `TfidfVectorizer` (N-Gram tingkat kata dengan rentang (1,2)).

### B. Python-CRFsuite (`python-crfsuite`)
* **Fungsi:** Pengenalan Entitas Bernama / Ekstraksi Slot Informasi (*Slot Filling / Sequence Labeling*).
* **Algoritma:** Conditional Random Fields (CRF) dengan Regularisasi L1/L2 ($c_1=0.1, c_2=0.1$).

### C. Regex & Standard Libraries (`re`, `json`, `pickle`)

## 3. Alur Aliran Data (Hulu ke Hilir)
Teks Input (Raw String) 
   ──> [normalizer.py] (Penghapusan kata noise & standardisasi singkatan)
   ──> [SVM Model]     (Deteksi Kategori Intent)
   ──> [CRF Model]     (Ekstraksi BIO-Tags tiap Kata)
   ──> [grounding_translator.py] (Resolusi konflik data, validasi tipe, pengikatan slot)
   ──> Output Akhir (Standardized JSON) ──> Siap dikonsumsi ROS / Perangkat Keras.""",

    "docs/system_guide/2_code_reference.md": """# Referensi Kode - Fungsi & Struktur Komponen Runtime

Dokumen ini menjelaskan fungsi teknis dari setiap skrip operasional yang berada di dalam direktori `src/core_pipeline/`.

## 1. `normalizer.py` (Gerbang Utama/Sanitasi Input)
* **Fungsi Utama:**
    * Memisahkan teks angka dan satuan yang menempel (contoh: `"10cm"` -> `"10 centimeter"`).
    * Mengonversi kata singkatan bahasa Indonesia ke baku menggunakan kamus `NORM_DICT`.
    * Mengeliminasi kata *noise* seperti `"woi"`, `"dong"`, `"heii"`.

## 2. `grounding_translator.py` (Interpreter & Sistem Pertahanan)
* **Aturan Tegas Terimplementasi:**
    1.  **Intent Supremacy:** Jika Intent adalah `STOP_COMMAND`, langsung paksa output menjadi tindakan darurat `STOP` (Rem Darurat).
    2.  **Deterministic Token-Unit Binding:** Menggunakan taksonomi biner untuk memisahkan satuan spasial (`METER`, `DERAJAT`) dan temporal (`MENIT`, `KALI`).
    3.  **Type Validation Fallback:** Menangkap eror konversi data numerik kualitatif dan memaksanya kembali ke nilai default aman (`1.0`).

## 3. `e2e_simulation.py` (Orchestrator Pipeline Akhir)
* **Fungsi Utama:** Mengintegrasikan model biner dari folder `build/`, menyatukan siklus pemanggilan dari hulu ke hilir.""",

    "docs/system_guide/3_deployment_guide.md": """# Panduan Penyebaran & Manajemen Aset Model Biner
""",
}
