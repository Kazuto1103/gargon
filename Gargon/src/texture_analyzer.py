"""
Surface Roughness Detection - Texture Feature Extraction and Comparison
Menggunakan GLCM (Gray-Level Co-occurrence Matrix) dan LBP (Local Binary Patterns)

Author: Nouzen
Date: 2026
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage import img_as_ubyte
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# 1. PREPROCESSING FUNCTION
# ============================================================================

def preprocess_image(image_path, target_size=None, verbose=True):
    """
    Membaca gambar dan melakukan preprocessing untuk analisis tekstur.
    
    Parameters:
    -----------
    image_path : str or Path
        Path ke file gambar yang akan diproses
    target_size : tuple, optional
        Ukuran target (height, width) untuk resize gambar.
        Jika None, gambar tidak akan di-resize.
    verbose : bool
        Jika True, akan menampilkan informasi gambar yang diproses
    
    Returns:
    --------
    img_gray : ndarray
        Gambar grayscale (uint8)
    """
    
    # Baca gambar dalam mode warna RGB
    img = cv2.imread(str(image_path))
    
    if img is None:
        raise FileNotFoundError(f"Gambar tidak ditemukan: {image_path}")
    
    # Konversi dari BGR ke RGB (OpenCV menggunakan BGR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Konversi ke grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Resize jika diperlukan (untuk komputasi yang lebih efisien)
    if target_size is not None:
        img_gray = cv2.resize(img_gray, (target_size[1], target_size[0]), 
                              interpolation=cv2.INTER_AREA)
        img_rgb = cv2.resize(img_rgb, (target_size[1], target_size[0]), 
                             interpolation=cv2.INTER_AREA)
    
    if verbose:
        print(f"✓ Gambar berhasil diproses: {Path(image_path).name}")
        print(f"  Ukuran: {img_gray.shape[0]} x {img_gray.shape[1]} pixels")
    
    return img_gray, img_rgb


# ============================================================================
# 2. GLCM FEATURE EXTRACTION
# ============================================================================

def extract_glcm_features(image_gray, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
    """
    Ekstraksi fitur tekstur menggunakan GLCM (Gray-Level Co-occurrence Matrix).
    
    GLCM menganalisis hubungan spasial antar pixel untuk mendeteksi tekstur.
    Fitur yang diekstrak: Contrast, Homogeneity, Energy, Correlation
    
    Parameters:
    -----------
    image_gray : ndarray
        Gambar grayscale (uint8)
    distances : list
        Jarak (d) untuk menghitung co-occurrence matrix
    angles : list
        Sudut orientasi dalam radian (0°, 45°, 90°, 135°)
    
    Returns:
    --------
    features_dict : dict
        Kamus berisi rata-rata nilai: Contrast, Homogeneity, Energy, Correlation
    glcm_matrix : ndarray
        GLCM matrix untuk keperluan visualisasi
    """
    
    # Pastikan gambar bertipe uint8
    image_gray = img_as_ubyte(image_gray)
    
    # Hitung GLCM dengan parameter yang ditentukan
    glcm = graycomatrix(image_gray, distances=distances, angles=angles, 
                        levels=256, symmetric=True, normed=True)
    
    # Ekstrak properti tekstur dari GLCM
    contrast = graycoprops(glcm, 'contrast').mean()
    homogeneity = graycoprops(glcm, 'homogeneity').mean()
    energy = graycoprops(glcm, 'energy').mean()
    correlation = graycoprops(glcm, 'correlation').mean()
    
    # Simpan dalam dictionary untuk kemudahan akses
    features_dict = {
        'Contrast': contrast,
        'Homogeneity': homogeneity,
        'Energy': energy,
        'Correlation': correlation
    }
    
    return features_dict, glcm


# ============================================================================
# 3. LBP FEATURE EXTRACTION
# ============================================================================

def extract_lbp_features(image_gray, radius=1, n_points=8, method='uniform'):
    """
    Ekstraksi fitur tekstur menggunakan LBP (Local Binary Patterns).
    
    LBP menghitung pola biner lokal di sekitar setiap pixel untuk mendeteksi tekstur.
    Histogram LBP yang dinormalisasi digunakan sebagai representasi fitur.
    
    Parameters:
    -----------
    image_gray : ndarray
        Gambar grayscale (uint8)
    radius : int
        Radius lingkaran untuk menghitung LBP
    n_points : int
        Jumlah sampling points di sekitar pixel pusat
    method : str
        Metode perhitungan LBP ('uniform', 'default', 'ror', 'var')
    
    Returns:
    --------
    lbp_hist : ndarray
        Histogram LBP yang dinormalisasi (fitur vektor)
    lbp_map : ndarray
        Peta LBP untuk visualisasi
    """
    
    # Pastikan gambar bertipe uint8
    image_gray = img_as_ubyte(image_gray)
    
    # Hitung LBP
    lbp_map = local_binary_pattern(image_gray, n_points, radius, method=method)
    
    # Hitung histogram dari LBP map dengan normalisasi
    lbp_hist, _ = np.histogram(lbp_map.ravel(), 
                               bins=np.arange(0, n_points + 3),
                               range=(0, n_points + 2))
    
    # Normalisasi histogram (sum = 1)
    lbp_hist = lbp_hist.astype(float) / lbp_hist.sum()
    
    # Hitung statistik dari histogram LBP
    lbp_mean = np.mean(lbp_map)
    lbp_std = np.std(lbp_map)
    
    return lbp_hist, lbp_map, lbp_mean, lbp_std


# ============================================================================
# 4. COMPARISON & VISUALIZATION PIPELINE
# ============================================================================

def compare_texture_features(image_path_1, image_path_2, 
                            target_size=(512, 512), 
                            image_labels=None):
    """
    Pipeline lengkap untuk membandingkan fitur tekstur dua gambar.
    
    Fungsi ini:
    1. Membaca dan preprocess dua gambar
    2. Ekstrak fitur GLCM dan LBP dari keduanya
    3. Tampilkan visualisasi perbandingan
    4. Cetak hasil analisis di terminal
    
    Parameters:
    -----------
    image_path_1 : str or Path
        Path gambar pertama (contoh: permukaan kasar)
    image_path_2 : str or Path
        Path gambar kedua (contoh: permukaan halus)
    target_size : tuple
        Ukuran target untuk resize gambar (height, width)
    image_labels : tuple, optional
        Label untuk kedua gambar (label_1, label_2)
        Default: ("Surface 1", "Surface 2")
    """
    
    if image_labels is None:
        image_labels = ("Surface 1", "Surface 2")
    
    print("\n" + "="*70)
    print("SURFACE ROUGHNESS TEXTURE ANALYSIS - GLCM vs LBP COMPARISON")
    print("="*70 + "\n")
    
    # ---- PREPROCESSING ----
    print("[1/4] Preprocessing gambar...")
    img_gray_1, img_rgb_1 = preprocess_image(image_path_1, target_size=target_size)
    img_gray_2, img_rgb_2 = preprocess_image(image_path_2, target_size=target_size)
    
    # ---- GLCM EXTRACTION ----
    print("\n[2/4] Ekstraksi fitur GLCM...")
    glcm_features_1, glcm_matrix_1 = extract_glcm_features(img_gray_1)
    glcm_features_2, glcm_matrix_2 = extract_glcm_features(img_gray_2)
    print("✓ GLCM features extracted from both images")
    
    # ---- LBP EXTRACTION ----
    print("\n[3/4] Ekstraksi fitur LBP...")
    lbp_hist_1, lbp_map_1, lbp_mean_1, lbp_std_1 = extract_lbp_features(img_gray_1)
    lbp_hist_2, lbp_map_2, lbp_mean_2, lbp_std_2 = extract_lbp_features(img_gray_2)
    print("✓ LBP features extracted from both images")
    
    # ---- VISUALIZATION ----
    print("\n[4/4] Membuat visualisasi...")
    create_comparison_visualization(
        img_rgb_1, img_rgb_2,
        lbp_map_1, lbp_map_2,
        glcm_features_1, glcm_features_2,
        lbp_mean_1, lbp_mean_2,
        lbp_std_1, lbp_std_2,
        image_labels
    )
    
    # ---- PRINT ANALYSIS RESULTS ----
    print_analysis_results(
        glcm_features_1, glcm_features_2,
        lbp_mean_1, lbp_mean_2,
        lbp_std_1, lbp_std_2,
        image_labels
    )
    
    print("\n" + "="*70)
    print("Analisis selesai!")
    print("="*70 + "\n")
    
    return {
        'image_1': {'gray': img_gray_1, 'rgb': img_rgb_1, 'label': image_labels[0]},
        'image_2': {'gray': img_gray_2, 'rgb': img_rgb_2, 'label': image_labels[1]},
        'glcm_features_1': glcm_features_1,
        'glcm_features_2': glcm_features_2,
        'lbp_hist_1': lbp_hist_1,
        'lbp_hist_2': lbp_hist_2
    }


# ============================================================================
# 5. VISUALIZATION HELPER FUNCTION
# ============================================================================

def create_comparison_visualization(img_rgb_1, img_rgb_2,
                                   lbp_map_1, lbp_map_2,
                                   glcm_1, glcm_2,
                                   lbp_mean_1, lbp_mean_2,
                                   lbp_std_1, lbp_std_2,
                                   labels):
    """
    Membuat visualisasi perbandingan untuk kedua gambar dengan subplot.
    
    Menampilkan:
    - Kolom kiri: Gambar asli
    - Kolom tengah: Peta LBP
    - Kolom kanan: Statistik tekstur
    """
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Surface Roughness Texture Analysis: GLCM vs LBP', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # ---- ROW 1: Surface 1 ----
    # Gambar asli
    axes[0, 0].imshow(img_rgb_1)
    axes[0, 0].set_title(f'{labels[0]}\n(Original Image)', fontweight='bold')
    axes[0, 0].axis('off')
    
    # Peta LBP
    lbp_display_1 = axes[0, 1].imshow(lbp_map_1, cmap='gray')
    axes[0, 1].set_title(f'{labels[0]}\n(LBP Map)', fontweight='bold')
    axes[0, 1].axis('off')
    plt.colorbar(lbp_display_1, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    # Statistik GLCM
    glcm_text_1 = f"""GLCM Features:
    
Contrast:      {glcm_1['Contrast']:.4f}
Homogeneity:   {glcm_1['Homogeneity']:.4f}
Energy:        {glcm_1['Energy']:.4f}
Correlation:   {glcm_1['Correlation']:.4f}

LBP Statistics:

Mean:          {lbp_mean_1:.4f}
Std Dev:       {lbp_std_1:.4f}"""
    
    axes[0, 2].text(0.05, 0.95, glcm_text_1, transform=axes[0, 2].transAxes,
                    fontsize=11, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[0, 2].axis('off')
    axes[0, 2].set_title(f'{labels[0]}\n(Feature Statistics)', fontweight='bold')
    
    # ---- ROW 2: Surface 2 ----
    # Gambar asli
    axes[1, 0].imshow(img_rgb_2)
    axes[1, 0].set_title(f'{labels[1]}\n(Original Image)', fontweight='bold')
    axes[1, 0].axis('off')
    
    # Peta LBP
    lbp_display_2 = axes[1, 1].imshow(lbp_map_2, cmap='gray')
    axes[1, 1].set_title(f'{labels[1]}\n(LBP Map)', fontweight='bold')
    axes[1, 1].axis('off')
    plt.colorbar(lbp_display_2, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    # Statistik GLCM
    glcm_text_2 = f"""GLCM Features:
    
Contrast:      {glcm_2['Contrast']:.4f}
Homogeneity:   {glcm_2['Homogeneity']:.4f}
Energy:        {glcm_2['Energy']:.4f}
Correlation:   {glcm_2['Correlation']:.4f}

LBP Statistics:

Mean:          {lbp_mean_2:.4f}
Std Dev:       {lbp_std_2:.4f}"""
    
    axes[1, 2].text(0.05, 0.95, glcm_text_2, transform=axes[1, 2].transAxes,
                    fontsize=11, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    axes[1, 2].axis('off')
    axes[1, 2].set_title(f'{labels[1]}\n(Feature Statistics)', fontweight='bold')
    
    plt.tight_layout()
    plt.show()


# ============================================================================
# 6. ANALYSIS RESULTS PRINTING
# ============================================================================

def print_analysis_results(glcm_1, glcm_2, lbp_mean_1, lbp_mean_2,
                          lbp_std_1, lbp_std_2, labels):
    """
    Mencetak hasil analisis perbandingan fitur tekstur ke terminal.
    
    Meliputi:
    - Perbandingan nilai fitur GLCM
    - Perbandingan statistik LBP
    - Analisis sensitivitas terhadap kekasaran permukaan
    """
    
    print("\n" + "="*70)
    print("DETAILED TEXTURE FEATURE COMPARISON")
    print("="*70)
    
    # ---- GLCM COMPARISON ----
    print(f"\n{'GLCM (Gray-Level Co-occurrence Matrix) Features':^70}")
    print("-"*70)
    print(f"{'Feature':<20} {labels[0]:<20} {labels[1]:<20} {'Difference':<10}")
    print("-"*70)
    
    for feature_name in ['Contrast', 'Homogeneity', 'Energy', 'Correlation']:
        val_1 = glcm_1[feature_name]
        val_2 = glcm_2[feature_name]
        diff = abs(val_1 - val_2)
        print(f"{feature_name:<20} {val_1:<20.6f} {val_2:<20.6f} {diff:<10.6f}")
    
    # ---- LBP COMPARISON ----
    print(f"\n{'LBP (Local Binary Patterns) Statistics':^70}")
    print("-"*70)
    print(f"{'Statistic':<20} {labels[0]:<20} {labels[1]:<20} {'Difference':<10}")
    print("-"*70)
    
    diff_mean = abs(lbp_mean_1 - lbp_mean_2)
    diff_std = abs(lbp_std_1 - lbp_std_2)
    
    print(f"{'Mean':<20} {lbp_mean_1:<20.6f} {lbp_mean_2:<20.6f} {diff_mean:<10.6f}")
    print(f"{'Std Deviation':<20} {lbp_std_1:<20.6f} {lbp_std_2:<20.6f} {diff_std:<10.6f}")
    
    # ---- SENSITIVITY ANALYSIS ----
    print(f"\n{'SENSITIVITY ANALYSIS - Which method is more sensitive?':^70}")
    print("-"*70)
    
    # GLCM sensitivity
    glcm_contrast_diff = abs(glcm_1['Contrast'] - glcm_2['Contrast'])
    glcm_homogeneity_diff = abs(glcm_1['Homogeneity'] - glcm_2['Homogeneity'])
    glcm_energy_diff = abs(glcm_1['Energy'] - glcm_2['Energy'])
    glcm_correlation_diff = abs(glcm_1['Correlation'] - glcm_2['Correlation'])
    
    print("\nGLCM Sensitivity (Perubahan fitur antar dua permukaan):")
    print(f"  • Contrast:      {glcm_contrast_diff:.6f} (Mendeteksi perbedaan kontras lokal)")
    print(f"  • Homogeneity:   {glcm_homogeneity_diff:.6f} (Mendeteksi uniformitas tekstur)")
    print(f"  • Energy:        {glcm_energy_diff:.6f} (Mendeteksi konsistensi pola)")
    print(f"  • Correlation:   {glcm_correlation_diff:.6f} (Mendeteksi hubungan spasial)")
    
    print("\nLBP Sensitivity (Perubahan pola biner lokal):")
    print(f"  • Mean LBP:      {diff_mean:.6f} (Perubahan rata-rata pola)")
    print(f"  • Std Dev LBP:   {diff_std:.6f} (Perubahan variasi pola)")
    
    # Determine which method is more sensitive
    glcm_max = max(glcm_contrast_diff, glcm_homogeneity_diff, 
                   glcm_energy_diff, glcm_correlation_diff)
    lbp_max = max(diff_mean, diff_std)
    
    print("\n" + "─"*70)
    if glcm_max > lbp_max * 1.2:
        print("✓ GLCM menunjukkan sensitivitas LEBIH TINGGI terhadap perbedaan kekasaran")
        print(f"  (GLCM max: {glcm_max:.6f} > LBP max: {lbp_max:.6f})")
        print("  Rekomendasi: Gunakan GLCM untuk analisis kekasaran permukaan yang lebih detail")
    elif lbp_max > glcm_max * 1.2:
        print("✓ LBP menunjukkan sensitivitas LEBIH TINGGI terhadap perbedaan kekasaran")
        print(f"  (LBP max: {lbp_max:.6f} > GLCM max: {glcm_max:.6f})")
        print("  Rekomendasi: Gunakan LBP untuk analisis kekasaran permukaan yang lebih responsive")
    else:
        print("✓ Kedua metode menunjukkan sensitivitas yang SEIMBANG")
        print(f"  (GLCM max: {glcm_max:.6f} ≈ LBP max: {lbp_max:.6f})")
        print("  Rekomendasi: Gabungkan GLCM dan LBP untuk hasil analisis yang lebih robust")
    print("─"*70)
    
    print("\n" + "="*70 + "\n")


# ============================================================================
# 7. MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    """
    Contoh penggunaan script untuk membandingkan dua gambar permukaan.
    
    Script akan mencari gambar dari folder 'data/roughness_kasar/' dan 'data/roughness_halus/'
    """
    
    # Tentukan path ke folder data
    data_folder = Path(__file__).parent.parent / "data"
    kasar_folder = data_folder / "roughness_kasar"
    halus_folder = data_folder / "roughness_halus"
    
    # Cari file gambar dengan dukungan format: .jpg, .png, .tif, .tiff
    image_files = []
    
    # Cek di folder roughness_kasar
    if kasar_folder.exists():
        image_files.extend(list(kasar_folder.glob("*.jpg")) + 
                          list(kasar_folder.glob("*.png")) + 
                          list(kasar_folder.glob("*.tif")) + 
                          list(kasar_folder.glob("*.tiff")) + 
                          list(kasar_folder.glob("*.JPG")) + 
                          list(kasar_folder.glob("*.PNG")) + 
                          list(kasar_folder.glob("*.TIF")) + 
                          list(kasar_folder.glob("*.TIFF")))
    
    # Cek di folder roughness_halus
    if halus_folder.exists():
        image_files.extend(list(halus_folder.glob("*.jpg")) + 
                          list(halus_folder.glob("*.png")) + 
                          list(halus_folder.glob("*.tif")) + 
                          list(halus_folder.glob("*.tiff")) + 
                          list(halus_folder.glob("*.JPG")) + 
                          list(halus_folder.glob("*.PNG")) + 
                          list(halus_folder.glob("*.TIF")) + 
                          list(halus_folder.glob("*.TIFF")))
    
    # Fallback: cari di folder data langsung
    if len(image_files) < 2:
        image_files = list(data_folder.glob("*.jpg")) + \
                      list(data_folder.glob("*.png")) + \
                      list(data_folder.glob("*.tif")) + \
                      list(data_folder.glob("*.tiff")) + \
                      list(data_folder.glob("*.JPG")) + \
                      list(data_folder.glob("*.PNG")) + \
                      list(data_folder.glob("*.TIF")) + \
                      list(data_folder.glob("*.TIFF"))
    
    image_files = sorted(set(image_files))  # Remove duplicates and sort
    
    if len(image_files) < 2:
        print("\n⚠️  Warning: Minimal 2 gambar harus ada di folder 'data/'")
        print(f"   Ditemukan hanya: {len(image_files)} gambar")
        print("\nPastikan folder 'data/' atau subfolder berikut berisi gambar:")
        print("   - data/roughness_kasar/ (Minimal 1 gambar permukaan kasar)")
        print("   - data/roughness_halus/ (Minimal 1 gambar permukaan halus)")
        print("\nFormat gambar yang didukung: .jpg, .png, .tif, .tiff")
    else:
        # Gunakan dua gambar pertama sebagai contoh
        image_path_1 = image_files[0]
        image_path_2 = image_files[1]
        
        # Jalankan analisis perbandingan
        results = compare_texture_features(
            image_path_1=image_path_1,
            image_path_2=image_path_2,
            target_size=(512, 512),
            image_labels=(
                image_files[0].stem,  # Nama file tanpa ekstensi
                image_files[1].stem
            )
        )
