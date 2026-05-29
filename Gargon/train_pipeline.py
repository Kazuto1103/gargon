"""
Surface Roughness Classification - Training Pipeline (Fase 1)
Skrip modular untuk mengekstrak fitur hibrida (GLCM + LBP), menyiapkan dataset,
dan melatih model RandomForestClassifier untuk mendeteksi kekasaran permukaan.

Author: Nouzen
Date: 2026
"""

import os
import shutil
import warnings
import numpy as np
import pandas as pd
import cv2
import joblib
from pathlib import Path

# Scikit-image imports
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage import img_as_ubyte

# Scikit-learn imports
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION & PATHS
# ============================================================================
PROJECT_ROOT = Path(__file__).parent
DATA_FOLDER = PROJECT_ROOT / "data"
KASAR_FOLDER = DATA_FOLDER / "kasar"
HALUS_FOLDER = DATA_FOLDER / "halus"
MODEL_PATH = PROJECT_ROOT / "model_kekasaran.pkl"


# ============================================================================
# 1. DATASET SETUP & GENERATION
# ============================================================================
def setup_data_directories(verbose=True):
    """
    Membuat folder data/kasar dan data/halus jika belum ada.
    Menyalin gambar asli ke data/kasar, dan menghasilkan gambar sintetis
    halus di data/halus menggunakan Gaussian Blur jika kosong.
    """
    KASAR_FOLDER.mkdir(parents=True, exist_ok=True)
    HALUS_FOLDER.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print("[SETUP] Memeriksa direktori dataset...")
        
    # 1. Cari gambar asli di folder data/ atau data/roughness_kasar/
    source_images = list(DATA_FOLDER.glob("*.tif")) + \
                    list(DATA_FOLDER.glob("*.tiff")) + \
                    list(DATA_FOLDER.glob("*.png")) + \
                    list(DATA_FOLDER.glob("*.jpg"))
    
    # Jika tidak ada di root data/, coba cari di data/roughness_kasar/
    if len(source_images) == 0:
        roughness_kasar_dir = DATA_FOLDER / "roughness_kasar"
        if roughness_kasar_dir.exists():
            source_images = list(roughness_kasar_dir.glob("*.tif")) + \
                            list(roughness_kasar_dir.glob("*.tiff"))
            
    if len(source_images) == 0:
        raise FileNotFoundError(
            "Tidak ada file gambar (.tif, .png, .jpg) yang ditemukan di folder data/ atau data/roughness_kasar/"
        )
        
    # 2. Salin gambar asli ke data/kasar
    kasar_images = list(KASAR_FOLDER.glob("*.tif")) + \
                   list(KASAR_FOLDER.glob("*.tiff")) + \
                   list(KASAR_FOLDER.glob("*.png")) + \
                   list(KASAR_FOLDER.glob("*.jpg"))
                   
    if len(kasar_images) == 0:
        if verbose:
            print(f"[SETUP] Menyalin {len(source_images)} gambar asli ke {KASAR_FOLDER.name}...")
        for img_path in source_images:
            shutil.copy2(img_path, KASAR_FOLDER / img_path.name)
        kasar_images = list(KASAR_FOLDER.glob("*"))
        
    # 3. Buat data halus sintetis jika data/halus kosong
    halus_images = list(HALUS_FOLDER.glob("*.tif")) + \
                   list(HALUS_FOLDER.glob("*.tiff")) + \
                   list(HALUS_FOLDER.glob("*.png")) + \
                   list(HALUS_FOLDER.glob("*.jpg"))
                   
    if len(halus_images) == 0:
        if verbose:
            print(f"[SETUP] Membuat data halus sintetis (Gaussian Blur) sebanyak {len(kasar_images)} gambar...")
        for img_path in kasar_images:
            # Baca gambar
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            # Terapkan Gaussian Blur yang kuat untuk menghilangkan tekstur permukaan kasar
            # kernel size (45, 45) memberikan efek blur halus yang signifikan
            smooth_img = cv2.GaussianBlur(img, (45, 45), 0)
            
            # Simpan dengan nama baru di folder halus
            dest_name = f"smooth_{img_path.name}"
            cv2.imwrite(str(HALUS_FOLDER / dest_name), smooth_img)
            
        halus_images = list(HALUS_FOLDER.glob("*"))
        
    if verbose:
        print("[SETUP] Dataset siap:")
        print(f"  - Kasar (Rough): {len(kasar_images)} gambar di {KASAR_FOLDER}")
        print(f"  - Halus (Smooth): {len(halus_images)} gambar di {HALUS_FOLDER}\n")
        

# ============================================================================
# 2. FEATURE EXTRACTION PIPELINE
# ============================================================================
def extract_hybrid_features(image_path, target_size=(256, 256)):
    """
    Mengekstrak fitur hibrida GLCM dan LBP dari satu berkas gambar.
    
    Parameters:
    -----------
    image_path : Path or str
        Path ke file gambar yang akan diekstrak fiturnya.
    target_size : tuple
        Dimensi resize untuk standardisasi ukuran gambar.
        
    Returns:
    --------
    features : ndarray
        1D Array berisi gabungan fitur GLCM dan histogram LBP.
    """
    # 1. Baca gambar dalam mode grayscale
    img_gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise ValueError(f"Gagal membaca gambar atau gambar korup: {image_path}")
        
    # 2. Preprocessing dasar: Standardisasi ukuran gambar (resize)
    img_gray = cv2.resize(img_gray, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA)
    
    # Pastikan data bertipe uint8 (ubyte) untuk skimage
    img_gray = img_as_ubyte(img_gray)
    
    # 3. Ekstraksi Fitur GLCM (Gray-Level Co-occurrence Matrix)
    # Jarak d=1, Sudut 0, 45, 90, 135 derajat dalam radian
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm = graycomatrix(img_gray, distances=[1], angles=angles, levels=256, symmetric=True, normed=True)
    
    # Ekstrak metrik statistik GLCM dan ambil rata-rata dari keempat sudut
    contrast = graycoprops(glcm, 'contrast').mean()
    homogeneity = graycoprops(glcm, 'homogeneity').mean()
    energy = graycoprops(glcm, 'energy').mean()
    correlation = graycoprops(glcm, 'correlation').mean()
    
    glcm_vector = np.array([contrast, homogeneity, energy, correlation])
    
    # 4. Ekstraksi Fitur LBP (Local Binary Patterns)
    # Parameter: P=8 sampling points, R=1 radius, method='uniform'
    n_points = 8
    radius = 1
    lbp_map = local_binary_pattern(img_gray, n_points, radius, method='uniform')
    
    # Hitung histogram dari peta LBP (uniform LBP menghasilkan P+2 bins)
    lbp_hist, _ = np.histogram(lbp_map.ravel(), 
                               bins=np.arange(0, n_points + 3),
                               range=(0, n_points + 2))
    
    # Normalisasi histogram (sum = 1)
    lbp_hist = lbp_hist.astype(float)
    lbp_hist /= (lbp_hist.sum() + 1e-10) # cegah pembagian dengan nol
    
    # 5. Gabungkan fitur statistik GLCM dan vektor histogram LBP (Feature Fusion)
    feature_vector = np.hstack([glcm_vector, lbp_hist])
    
    return feature_vector


# ============================================================================
# 3. DATASET PREPARATION
# ============================================================================
def prepare_dataset(verbose=True):
    """
    Membaca seluruh gambar dari folder kasar dan halus,
    mengekstrak fitur hibrida, serta membuat matriks X dan label y.
    
    Returns:
    --------
    X : ndarray
        Matriks fitur berukuran (N, D)
    y : ndarray
        Array label biner (N,) -> 0: halus, 1: kasar
    """
    X = []
    y = []
    
    # Cari semua gambar di folder kasar dan halus
    kasar_paths = sorted([p for p in KASAR_FOLDER.glob("*") if p.suffix.lower() in ['.tif', '.tiff', '.png', '.jpg', '.jpeg']])
    halus_paths = sorted([p for p in HALUS_FOLDER.glob("*") if p.suffix.lower() in ['.tif', '.tiff', '.png', '.jpg', '.jpeg']])
    
    if verbose:
        print("[DATA] Mengekstraksi fitur dari dataset...")
        
    # Ekstraksi untuk kelas kasar (Label = 1)
    for path in kasar_paths:
        try:
            feats = extract_hybrid_features(path)
            X.append(feats)
            y.append(1)
        except Exception as e:
            print(f"  [WARN] Gagal memproses {path.name}: {str(e)}")
            
    # Ekstraksi untuk kelas halus (Label = 0)
    for path in halus_paths:
        try:
            feats = extract_hybrid_features(path)
            X.append(feats)
            y.append(0)
        except Exception as e:
            print(f"  [WARN] Gagal memproses {path.name}: {str(e)}")
            
    X = np.array(X)
    y = np.array(y)
    
    if verbose:
        print(f"[DATA] Ekstraksi selesai. Total sampel: {len(X)}")
        print(f"  - Dimensi Matriks Fitur X: {X.shape}")
        print(f"  - Distribusi Kelas y: Kasar={np.sum(y == 1)}, Halus={np.sum(y == 0)}")
        print()
        
    return X, y


# ============================================================================
# 4. TRAINING & EVALUATION
# ============================================================================
def train_and_evaluate_model(X, y):
    """
    Membagi dataset menjadi train/test (80:20), melatih model RandomForest,
    dan mengevaluasi performanya menggunakan akurasi serta classification report.
    
    Returns:
    --------
    model : RandomForestClassifier
        Model terlatih.
    """
    print("[TRAINING] Membagi dataset menjadi training (80%) dan testing (20%)...")
    # Membagi data dengan stratify=y untuk memastikan proporsi kelas seimbang di train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("[TRAINING] Melatih model RandomForestClassifier...")
    # Inisialisasi model RandomForest
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Pengujian model
    y_pred = model.predict(X_test)
    
    # Evaluasi metrik
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Halus (0)', 'Kasar (1)'])
    
    print("\n" + "="*60)
    print("METRIK EVALUASI MODEL (RANDOM FOREST)")
    print("="*60)
    print(f"Akurasi Pengujian (Accuracy Score): {accuracy:.2%}")
    print("\nClassification Report:")
    print(report)
    print("="*60 + "\n")
    
    return model


# ============================================================================
# 5. MODEL SERIALIZATION
# ============================================================================
def save_model(model, filepath):
    """
    Menyimpan objek model terlatih ke dalam file biner (.pkl).
    """
    print(f"[SERIALISASI] Menyimpan model ke '{filepath.name}'...")
    joblib.dump(model, filepath)
    print(f"[SUCCESS] File model '{filepath.name}' berhasil disimpan!")
    print("  Otak model siap digunakan untuk live inference di Fase berikutnya.\n")


# ============================================================================
# MAIN PIPELINE EXECUTION
# ============================================================================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("SURFACE ROUGHNESS DETECTOR - OFFLINE TRAINING PIPELINE")
    print("="*70 + "\n")
    
    try:
        # Langkah 1: Setup direktori dan file dataset
        setup_data_directories(verbose=True)
        
        # Langkah 2: Siapkan dataset (Ekstraksi fitur hibrida GLCM+LBP)
        X, y = prepare_dataset(verbose=True)
        
        # Langkah 3: Latih dan evaluasi model RandomForest
        model = train_and_evaluate_model(X, y)
        
        # Langkah 4: Simpan model terlatih
        save_model(model, MODEL_PATH)
        
        print("="*70)
        print("[SUCCESS] PIPELINE TRAINING SELESAI DENGAN SUKSES!")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n[ERROR] Terjadi kesalahan dalam pipeline training: {str(e)}")
        import traceback
        traceback.print_exc()
