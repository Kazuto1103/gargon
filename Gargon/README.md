# Surface Roughness Detection / Texture Analysis

Proyek Computer Vision untuk mendeteksi dan menganalisis kekasaran permukaan menggunakan teknik pemrosesan citra lanjutan.

## Struktur Proyek

```
Gargon/
├── data/              # Tempat menyimpan gambar sampel (kasar & halus)
├── src/               # Script utama untuk analisis
├── requirements.txt   # Dependency Python
└── README.md          # Dokumentasi proyek
```

## Setup Lingkungan Kerja

### 1. Buat Virtual Environment

**Windows (Command Prompt atau PowerShell):**
```bash
python -m venv venv
```

### 2. Aktifkan Virtual Environment

**Windows - Command Prompt:**
```bash
venv\Scripts\activate.bat
```

**Windows - PowerShell:**
```bash
venv\Scripts\Activate.ps1
```

Jika mendapat error di PowerShell, jalankan ini terlebih dahulu:
```bash
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 3. Upgrade pip (Opsional tapi disarankan)

```bash
python -m pip install --upgrade pip
```

### 4. Instal Semua Dependencies

```bash
pip install -r requirements.txt
```

## Library yang Digunakan

| Library | Fungsi |
|---------|--------|
| **opencv-python** | Pemrosesan citra (baca, resize, filter) |
| **scikit-image** | GLCM, LBP, dan teknik analisis tekstur |
| **numpy** | Operasi array dan kalkulasi numerik |
| **matplotlib** | Visualisasi hasil analisis |
| **scikit-learn** | Normalisasi data dan model ML (regresi, klasifikasi) |
| **pillow** | Manipulasi citra tambahan |
| **scipy** | Kalkulasi statistik dan pemrosesan sinyal |

## Verifikasi Instalasi

Setelah mengaktifkan venv dan install dependencies, cek instalasi dengan:

```bash
python -c "import cv2; import numpy; import matplotlib; import skimage; import sklearn; print('✓ Semua library berhasil diinstall!')"
```

## Deaktivasi Virtual Environment

Ketika selesai bekerja, deaktivasi venv dengan:

```bash
deactivate
```

## Tips Pengembangan

1. **Data Sampling**: Letakkan gambar kasar dan halus di folder `data/`
2. **Modularitas**: Buat script terpisah di `src/` untuk setiap tahap analisis
3. **Versionning**: Simpan requirements.txt di git untuk konsistensi environment
4. **Testing**: Buat script test sederhana untuk validasi pipeline

---

Selamat mengembangkan! 🚀
