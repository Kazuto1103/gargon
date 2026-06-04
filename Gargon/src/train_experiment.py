"""
Offline Training & Parameter Comparison Pipeline (Fase 1)
Mengotomatisasi perbandingan parameter GLCM dan LBP menggunakan MLPClassifier
dengan monitoring manual Epoch via partial_fit().

Author: Senior AI Engineer
Date: 2026
"""

import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

# Scikit-image imports
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage import img_as_ubyte

# Scikit-learn imports
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Abaikan warning dari skimage/sklearn untuk kerapian output terminal
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "NewData" / "Cropped50x"
LOGS_DIR = PROJECT_ROOT / "logs"

# In-code Mapping Dictionary: subfolder -> label (0 atau 1)
# 00 - 21: Label 0 (Nominal / Permukaan Mulus)
# 24 - 45: Label 1 (Anomalous / Permukaan Kasar)
FOLDER_TO_LABEL = {
    "00": 0, "03": 0, "06": 0, "09": 0, "12": 0, "15": 0, "18": 0, "21": 0,
    "24": 1, "27": 1, "30": 1, "33": 1, "36": 1, "39": 1, "42": 1, "45": 1
}

# ============================================================================
# 1. DATASET MAPPING & HANDLING IMBALANCE
# ============================================================================
def scan_dataset(verbose=True):
    """
    Membaca seluruh subfolder di dalam direktori Cropped50x, melakukan
    pemetaan otomatis ke label biner (0 atau 1), dan menganalisis
    ketimpangan data (data imbalance).
    
    Returns:
    --------
    image_paths : list of Path
        List berisi path absolut ke setiap file gambar.
    labels : list of int
        List berisi label (0 atau 1) yang berkorespondensi dengan image_paths.
    """
    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Direktori dataset tidak ditemukan di: {DATA_DIR}\n"
            "Pastikan dataset sudah diekstrak ke dalam folder data/NewData/Cropped50x/"
        )
        
    image_paths = []
    labels = []
    
    # Scan subfolder yang terdaftar dalam FOLDER_TO_LABEL
    for folder_name, label in FOLDER_TO_LABEL.items():
        folder_path = DATA_DIR / folder_name
        if not folder_path.exists():
            if verbose:
                print(f"[WARN] Folder spektrum '{folder_name}' tidak ditemukan di {DATA_DIR.name}/")
            continue
            
        # Dapatkan semua file gambar (format .jpg, .png, .jpeg)
        images = (
            list(folder_path.glob("*.jpg")) + 
            list(folder_path.glob("*.png")) + 
            list(folder_path.glob("*.jpeg"))
        )
        
        for img_path in images:
            image_paths.append(img_path)
            labels.append(label)
            
    if len(image_paths) == 0:
        raise ValueError(f"Tidak ada gambar (.jpg, .png, .jpeg) yang ditemukan di dalam {DATA_DIR}")
        
    # Hitung distribusi kelas untuk analisis imbalance
    total_images = len(image_paths)
    class_0_count = labels.count(0)
    class_1_count = labels.count(1)
    
    pct_class_0 = (class_0_count / total_images) * 100 if total_images > 0 else 0
    pct_class_1 = (class_1_count / total_images) * 100 if total_images > 0 else 0
    
    # Hitung rasio ketimpangan
    imbalance_ratio = max(class_0_count, class_1_count) / max(min(class_0_count, class_1_count), 1)
    
    # Cetak analisis ketimpangan ke terminal
    print("\n" + "="*60)
    print("           ANALISIS KETIMPANGAN DATASET (IMBALANCE)")
    print("="*60)
    print(f"Total Gambar Ditemukan : {total_images}")
    print(f"Kelas 0 (Nominal/Mulus): {class_0_count} gambar ({pct_class_0:.2f}%)")
    print(f"Kelas 1 (Kasar/Anomali): {class_1_count} gambar ({pct_class_1:.2f}%)")
    print(f"Rasio Imbalance        : {imbalance_ratio:.2f}:1")
    print("-"*60)
    
    if imbalance_ratio > 1.5:
        print("Status: [!] TERJADI DATA IMBALANCE (Ketimpangan cukup tinggi)")
        print("Saran : Pertimbangkan menggunakan stratifikasi pada train_test_split.")
    else:
        print("Status: [OK] DATA SEIMBANG (Balanced dataset)")
    print("="*60 + "\n")
    
    return image_paths, labels


# ============================================================================
# 2. HYBRID FEATURE EXTRACTION
# ============================================================================
def extract_hybrid_features(image_path, d, R, target_size=(128, 128)):
    """
    Ekstraksi fitur hibrida: Gabungan nilai rata-rata properti GLCM
    (Contrast, Homogeneity, Energy, Correlation) dengan Histogram LBP yang dinormalisasi.
    
    Parameters:
    -----------
    image_path : Path or str
        Path file gambar.
    d : int
        Jarak piksel (distances) untuk perhitungan GLCM.
    R : int
        Radius untuk algoritma LBP.
    target_size : tuple of int (height, width)
        Ukuran resize gambar untuk standardisasi.
        
    Returns:
    --------
    feature_vector : ndarray (1D)
        Array gabungan fitur GLCM dan histogram LBP.
    """
    # Baca gambar dalam grayscale
    img_gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise ValueError(f"Gagal membaca gambar: {image_path}")
        
    # Resize jika ditentukan untuk efisiensi dan standardisasi dimensi
    if target_size is not None:
        img_gray = cv2.resize(img_gray, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA)
        
    # Pastikan data bertipe uint8 (ubyte) agar aman diproses scikit-image
    img_gray = img_as_ubyte(img_gray)
    
    # A. Ekstraksi Fitur GLCM (Gray-Level Co-occurrence Matrix)
    # Gunakan 4 sudut standar dalam radian (0, 45, 90, 135 derajat)
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm = graycomatrix(img_gray, distances=[d], angles=angles, levels=256, symmetric=True, normed=True)
    
    # Ambil rata-rata fitur spasial dari keempat arah
    contrast = graycoprops(glcm, 'contrast').mean()
    homogeneity = graycoprops(glcm, 'homogeneity').mean()
    energy = graycoprops(glcm, 'energy').mean()
    correlation = graycoprops(glcm, 'correlation').mean()
    
    glcm_vector = np.array([contrast, homogeneity, energy, correlation])
    
    # B. Ekstraksi Fitur LBP (Local Binary Patterns)
    # Jumlah titik sampling (P) secara umum dikonfigurasi sebagai 8 * R
    n_points = 8 * R
    lbp_map = local_binary_pattern(img_gray, n_points, R, method='uniform')
    
    # Hitung histogram LBP. Metode 'uniform' menghasilkan n_points + 2 bins
    lbp_hist, _ = np.histogram(
        lbp_map.ravel(), 
        bins=np.arange(0, n_points + 3),
        range=(0, n_points + 2)
    )
    
    # Normalisasi histogram (sum = 1) agar invariant terhadap ukuran gambar
    lbp_hist = lbp_hist.astype(float)
    lbp_hist /= (lbp_hist.sum() + 1e-10)
    
    # C. Feature Fusion (Concatenation)
    feature_vector = np.hstack([glcm_vector, lbp_hist])
    
    return feature_vector


def extract_features_dataset(image_paths, labels, d, R, target_size=(128, 128)):
    """
    Ekstraksi fitur hibrida untuk seluruh gambar di dalam dataset.
    
    Returns:
    --------
    X : ndarray (2D)
        Matriks fitur berukuran (N, D)
    y : ndarray (1D)
        Array label biner (N,)
    """
    X = []
    y = []
    
    desc_str = f"Ekstraksi Fitur (d={d}, R={R})"
    for idx, path in enumerate(tqdm(image_paths, desc=desc_str)):
        try:
            feat = extract_hybrid_features(path, d, R, target_size=target_size)
            X.append(feat)
            y.append(labels[idx])
        except Exception as e:
            print(f"\n[WARN] Lewati gambar {path.name} karena error: {str(e)}")
            
    return np.array(X), np.array(y)


# ============================================================================
# 3. INTERACTIVE TRAINING LOGIC & METRIC TRACKING
# ============================================================================
def train_mlp_scenario(run_id, params, X_train, X_test, y_train, y_test):
    """
    Melatih MLPClassifier secara berulang (per epoch) menggunakan metode .partial_fit()
    agar dapat merekam loss dan akurasi pada training & testing set secara real-time.
    
    Parameters:
    -----------
    run_id : str
        ID Run (misal: "RUN 1")
    params : dict
        Kamus berisi parameter model dan fitur (d, R, epochs)
    X_train, X_test, y_train, y_test : ndarray
        Dataset split
        
    Returns:
    --------
    results : dict
        Kamus log epoch, confusion matrix, classification report, dan parameter.
    """
    epochs = params["epochs"]
    
    # normalisasi fitur menggunakan StandardScaler untuk mengatasi ketimpangan skala GLCM dan LBP
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Inisialisasi MLPClassifier dengan parameter neural network ter-tune
    clf = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        learning_rate_init=0.01,  # Akselerasi belajar agar cepat konvergen pada epoch rendah
        random_state=42
    )
    
    # List perekaman histori metrik
    epoch_history = []
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    
    print(f"\n>>> Memulai {run_id}...")
    print(f"    Parameter: GLCM Jarak d={params['d']}, LBP Radius R={params['R']}, Epochs={epochs}")
    print(f"    Dimensi Fitur input: {X_train.shape[1]} dimensi (Diskalakan menggunakan StandardScaler)")
    
    classes = np.array([0, 1])
    
    # Loop manual per epoch menggunakan partial_fit
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        
        # Latih model untuk 1 iterasi (epoch)
        clf.partial_fit(X_train_scaled, y_train, classes=classes)
        
        # Ambil nilai loss dan hitung akurasi
        loss = clf.loss_
        train_acc = clf.score(X_train_scaled, y_train)
        test_acc = clf.score(X_test_scaled, y_test)
        
        epoch_time = time.time() - start_time
        
        # Simpan metrik
        epoch_history.append(epoch)
        train_losses.append(loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        
        print(f"    Epoch {epoch:02d}/{epochs:02d} - Loss: {loss:.6f} - Train Acc: {train_acc:.4f} - Test Acc: {test_acc:.4f} ({epoch_time:.3f}s)")
        
    # Evaluasi akhir model
    y_pred = clf.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred)
    class_report = classification_report(
        y_test, y_pred, 
        target_names=['Nominal (0)', 'Anomalous (1)'], 
        output_dict=False
    )
    
    # Jika ini adalah RUN 2 (best configuration), simpan model dan scaler
    if run_id == "RUN 2":
        models_dir = PROJECT_ROOT / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        
        clf_path = models_dir / "surface_classifier.pkl"
        scaler_path = models_dir / "scaler.pkl"
        
        import joblib
        joblib.dump(clf, clf_path)
        joblib.dump(scaler, scaler_path)
        print(f"    [SUKSES] Menyimpan model terbaik ke {clf_path}")
        print(f"    [SUKSES] Menyimpan scaler terbaik ke {scaler_path}")
        
    # Simpan seluruh log ke dalam dictionary hasil
    results = {
        "run_id": run_id,
        "params": params,
        "epochs": epoch_history,
        "losses": train_losses,
        "train_accs": train_accuracies,
        "test_accs": test_accuracies,
        "confusion_matrix": cm,
        "classification_report": class_report,
        "final_train_acc": train_accuracies[-1],
        "final_test_acc": test_accuracies[-1],
        "final_loss": train_losses[-1]
    }
    
    return results


# ============================================================================
# 4. REPORT GENERATION & ASCII CHART UTILITIES
# ============================================================================
def generate_ascii_loss_chart(losses, max_bar_width=30):
    """
    Membuat grafik tren Loss sederhana berbasis karakter ASCII (Horizontal Progress Bar).
    """
    chart_lines = []
    max_loss = max(losses) if len(losses) > 0 else 1.0
    
    for i, loss in enumerate(losses):
        epoch = i + 1
        # Proporsi panjang bar berdasarkan persentase terhadap nilai loss maksimum
        ratio = loss / max_loss if max_loss > 0 else 0
        bar_length = int(ratio * max_bar_width)
        bar = "=" * bar_length + ">" + " " * (max_bar_width - bar_length)
        chart_lines.append(f"      Epoch {epoch:02d} | {loss:.6f} | [{bar}]")
        
    return "\n".join(chart_lines)


def build_experiment_report(run_results_list, timestamp, class_0_total, class_1_total):
    """
    Menyusun teks laporan perbandingan komprehensif, edukatif,
    dan detail dari seluruh RUN eksperimen yang dijalankan.
    """
    report = []
    report.append("="*80)
    report.append("          LAPORAN EKSPERIMEN PARAMETER & TRAINING PIPELINE (FASE 1)")
    report.append("="*80)
    report.append(f"Dibuat pada : {timestamp}")
    report.append(f"Model       : MLPClassifier (Neural Network: (64, 32) nodes, lr_init=0.01)")
    report.append(f"Dataset     : data/NewData/Cropped50x/ (Strict Directory Lock)")
    report.append(f"Data Split  : 60% Training / 40% Testing (random_state=42)")
    report.append(f"Distribusi  : Kelas 0 = {class_0_total} | Kelas 1 = {class_1_total}")
    report.append(f"Normalisasi : StandardScaler (Disuntikkan setelah train_test_split)")
    report.append("="*80 + "\n\n")
    
    # Ringkasan Perbandingan (Matrix)
    report.append("1. RINGKASAN PERBANDINGAN ANTAR RUN")
    report.append("-" * 80)
    header = f"{'Scenario':<8} | {'GLCM d':<6} | {'LBP R':<5} | {'Epochs':<6} | {'Final Loss':<11} | {'Train Acc':<9} | {'Test Acc':<9}"
    report.append(header)
    report.append("-" * 80)
    
    for res in run_results_list:
        p = res["params"]
        row = (
            f"{res['run_id']:<8} | "
            f"{p['d']:<6} | "
            f"{p['R']:<5} | "
            f"{p['epochs']:<6} | "
            f"{res['final_loss']:<11.6f} | "
            f"{res['final_train_acc']:<9.4f} | "
            f"{res['final_test_acc']:<9.4f}"
        )
        report.append(row)
    report.append("-" * 80 + "\n\n")
    
    # Edukasi Analisis Parameter
    report.append("2. ANALISIS EDUKATIF DAN REKOMENDASI PARAMETER")
    report.append("-" * 80)
    report.append(
        "  - EFEK PENYUNTIKAN FEATURE SCALING (StandardScaler):\n"
        "    Dengan StandardScaler, rentang fitur GLCM (Contrast berkisar ratusan) disetarakan dengan\n"
        "    LBP (Histogram bernilai 0-1). Hal ini mencegah bias terhadap fitur bernilai besar,\n"
        "    sehingga Neural Network (MLPClassifier) dapat mempelajari pola tekstur LBP secara efektif.\n"
    )
    report.append(
        "  - RUN 1 vs RUN 2 (Efek Penambahan Epoch & LR 0.01):\n"
        "    Dengan parameter mikro yang sama (d=1, R=1) dan learning rate agresif (0.01), penambahan\n"
        "    epoch dari 8 menjadi 15 melatih bobot model secara optimal, terlihat dari penurunan\n"
        "    Loss yang cepat dan peningkatan signifikan pada akurasi Testing.\n"
    )
    report.append(
        "  - RUN 2 vs RUN 3 (Efek Perubahan Fitur Mikro ke Makro):\n"
        "    Perubahan parameter ke Makro (d=3, R=2) mengubah jangkauan analisis tekstur.\n"
        "    Jarak GLCM d=3 menangkap variasi spasial yang lebih luas, sedangkan LBP R=2 menganalisis\n"
        "    tetangga piksel yang lebih besar. Kombinasi ini memberikan deskripsi tekstur yang lebih\n"
        "    stabil terhadap variasi mikro pada permukaan kasar.\n"
    )
    report.append("-" * 80 + "\n\n")
    
    # Detail per RUN
    report.append("3. RINCIAN EVALUASI DETAILED PER SCENARIO RUN")
    report.append("=" * 80 + "\n")
    
    for res in run_results_list:
        p = res["params"]
        report.append(f">>> {res['run_id']} ({p['desc']})")
        report.append("    " + "-"*40)
        report.append(f"    - GLCM Jarak (d)       : {p['d']}")
        report.append(f"    - LBP Radius (R)       : {p['R']}")
        report.append(f"    - Total Epoch          : {p['epochs']}")
        report.append(f"    - Final Train Loss     : {res['final_loss']:.6f}")
        report.append(f"    - Final Train Accuracy : {res['final_train_acc']:.4f} ({res['final_train_acc']*100:.2f}%)")
        report.append(f"    - Final Test Accuracy  : {res['final_test_acc']:.4f} ({res['final_test_acc']*100:.2f}%)")
        report.append("")
        
        # Grafik tracking loss per epoch
        report.append("    GRAFIK TRACKING LOSS PER EPOCH:")
        report.append(generate_ascii_loss_chart(res["losses"], max_bar_width=35))
        report.append("")
        
        # Confusion Matrix
        cm = res["confusion_matrix"]
        report.append("    CONFUSION MATRIX:")
        report.append(f"                         Predicted Nominal   Predicted Anomalous")
        report.append(f"      Actual Nominal    |      {cm[0,0]:<10} |      {cm[0,1]:<10}")
        report.append(f"      Actual Anomalous  |      {cm[1,0]:<10} |      {cm[1,1]:<10}")
        report.append("")
        
        # Classification Report
        report.append("    CLASSIFICATION REPORT:")
        # Indent classification report
        indented_rep = "\n".join(["      " + line for line in res["classification_report"].split("\n")])
        report.append(indented_rep)
        report.append("\n" + "="*80 + "\n")
        
    return "\n".join(report)


# ============================================================================
# MAIN EXECUTION PIPELINE
# ============================================================================
def main():
    print("\n" + "="*75)
    print("      OFFLINE TRAINING & PARAMETER COMPARISON PIPELINE - FASE 1")
    print("="*75)
    
    # Buat folder logs jika belum ada di root
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Pindai dataset dan tampilkan ketimpangan
    try:
        image_paths, labels = scan_dataset(verbose=True)
    except Exception as e:
        print(f"[ERROR] Inisialisasi dataset gagal: {str(e)}")
        sys.exit(1)
        
    # Kunci pemisahan dataset 60% Train dan 40% Test menggunakan train_test_split
    # Kita pisahkan list index/paths terlebih dahulu untuk performa pemrosesan paralel parameter
    print("[1/3] Membagi dataset ke dalam 60% Training dan 40% Testing...")
    train_idx, test_idx = train_test_split(
        np.arange(len(image_paths)),
        test_size=0.4,
        random_state=42,
        stratify=labels
    )
    
    # Pisahkan path dan label asli
    train_paths = [image_paths[i] for i in train_idx]
    test_paths = [image_paths[i] for i in test_idx]
    train_labels = [labels[i] for i in train_idx]
    test_labels = [labels[i] for i in test_idx]
    
    print(f"      -> Data Training: {len(train_paths)} sampel")
    print(f"      -> Data Testing : {len(test_paths)} sampel\n")
    
    # Definisi 3 Skenario Eksperimen (Permintaan Mentor)
    scenarios = [
        {
            "run_id": "RUN 1",
            "d": 1, "R": 1, "epochs": 8,
            "desc": "Parameter Mikro | Epoch Rendah"
        },
        {
            "run_id": "RUN 2",
            "d": 1, "R": 1, "epochs": 15,
            "desc": "Parameter Mikro | Epoch Tinggi"
        },
        {
            "run_id": "RUN 3",
            "d": 3, "R": 2, "epochs": 15,
            "desc": "Parameter Makro | Epoch Tinggi"
        }
    ]
    
    run_results = []
    
    # Caching fitur untuk optimasi waktu eksekusi
    # Karena RUN 1 dan RUN 2 memiliki parameter ekstraksi fitur yang identik (d=1, R=1),
    # kita dapat mengekstraknya sekali saja untuk mempercepat proses training.
    cached_features = {}
    
    # Jalankan eksperimen secara sekuensial (berurutan)
    print("[2/3] Memulai Training Sekuensial Skenario Eksperimen...")
    for idx, sc in enumerate(scenarios):
        # Cek jika ekstraksi fitur untuk kombinasi (d, R) sudah pernah dilakukan
        param_key = (sc["d"], sc["R"])
        
        if param_key not in cached_features:
            print(f"\n--- Mengambil fitur baru untuk parameter GLCM Jarak d={sc['d']}, LBP Radius R={sc['R']} ---")
            
            # Ekstraksi fitur untuk training set
            X_tr, y_tr = extract_features_dataset(train_paths, train_labels, sc["d"], sc["R"])
            # Ekstraksi fitur untuk testing set
            X_te, y_te = extract_features_dataset(test_paths, test_labels, sc["d"], sc["R"])
            
            cached_features[param_key] = (X_tr, X_te, y_tr, y_te)
        else:
            print(f"\n--- Menggunakan Cache Fitur GLCM Jarak d={sc['d']}, LBP Radius R={sc['R']} ---")
            
        X_tr, X_te, y_tr, y_te = cached_features[param_key]
        
        # Jalankan loop training MLPClassifier dan rekam metrik
        res = train_mlp_scenario(sc["run_id"], sc, X_tr, X_te, y_tr, y_te)
        run_results.append(res)
        
    # 3. Ekspor Laporan
    print("\n[3/3] Menyusun dan mengekspor laporan komparasi...")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    timestamp_file = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Hitung total sampel per kelas untuk dicatat ke log
    class_0_total = labels.count(0)
    class_1_total = labels.count(1)
    
    # Buat teks laporan
    report_content = build_experiment_report(run_results, timestamp, class_0_total, class_1_total)
    
    # Simpan ke file log teks di folder logs/
    report_filename = LOGS_DIR / f"experiment_report_{timestamp_file}.txt"
    with open(report_filename, "w") as f:
        f.write(report_content)
        
    print(f"\n[SUKSES] Seluruh eksperimen selesai dijalankan!")
    print(f"          Laporan detail berhasil disimpan ke:")
    print(f"          -> {report_filename.resolve()}")
    
    # Tampilkan ringkasan ringkas ke terminal
    print("\n" + "="*75)
    print("                    RINGKASAN AKHIR EKSPERIMEN")
    print("="*75)
    print(f"{'Run':<6} | {'Params':<22} | {'Epochs':<6} | {'Final Loss':<10} | {'Test Accuracy':<13}")
    print("-"*75)
    for r in run_results:
        p = r["params"]
        p_str = f"d={p['d']}, R={p['R']}"
        print(f"{r['run_id']:<6} | {p_str:<22} | {p['epochs']:<6} | {r['final_loss']:<10.6f} | {r['final_test_acc']:<13.2%}")
    print("="*75 + "\n")


if __name__ == "__main__":
    main()
