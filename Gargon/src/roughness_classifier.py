"""
Roughness Surface Classifier
Script untuk menganalisis, mengklasifikasi, dan memisahkan gambar permukaan
berdasarkan tingkat kekasaran (kasar vs halus) menggunakan GLCM dan LBP

Author: Nouzen
Date: 2026
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage import img_as_ubyte
import json
from tqdm import tqdm
import shutil
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION & PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_FOLDER = PROJECT_ROOT / "data"
ROUGHNESS_KASAR_FOLDER = DATA_FOLDER / "roughness_kasar"
ROUGHNESS_HALUS_FOLDER = DATA_FOLDER / "roughness_halus"
ANALYSIS_OUTPUT_FOLDER = PROJECT_ROOT / "analysis_results"

# Buat folder output jika belum ada
ANALYSIS_OUTPUT_FOLDER.mkdir(exist_ok=True)

# File hasil analisis
ANALYSIS_CSV = ANALYSIS_OUTPUT_FOLDER / "texture_analysis_results.csv"
ANALYSIS_JSON = ANALYSIS_OUTPUT_FOLDER / "texture_analysis_detailed.json"
CLASSIFICATION_REPORT = ANALYSIS_OUTPUT_FOLDER / "classification_report.txt"


# ============================================================================
# 1. FEATURE EXTRACTION FUNCTIONS (dari texture_analyzer.py)
# ============================================================================

def extract_glcm_features(image_gray, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
    """Ekstraksi fitur GLCM"""
    image_gray = img_as_ubyte(image_gray)
    glcm = graycomatrix(image_gray, distances=distances, angles=angles, 
                        levels=256, symmetric=True, normed=True)
    
    contrast = graycoprops(glcm, 'contrast').mean()
    homogeneity = graycoprops(glcm, 'homogeneity').mean()
    energy = graycoprops(glcm, 'energy').mean()
    correlation = graycoprops(glcm, 'correlation').mean()
    
    return {
        'Contrast': contrast,
        'Homogeneity': homogeneity,
        'Energy': energy,
        'Correlation': correlation
    }


def extract_lbp_features(image_gray, radius=1, n_points=8, method='uniform'):
    """Ekstraksi fitur LBP"""
    image_gray = img_as_ubyte(image_gray)
    lbp_map = local_binary_pattern(image_gray, n_points, radius, method=method)
    
    lbp_hist, _ = np.histogram(lbp_map.ravel(), 
                               bins=np.arange(0, n_points + 3),
                               range=(0, n_points + 2))
    lbp_hist = lbp_hist.astype(float) / lbp_hist.sum()
    
    lbp_mean = np.mean(lbp_map)
    lbp_std = np.std(lbp_map)
    lbp_entropy = -np.sum(lbp_hist[lbp_hist > 0] * np.log2(lbp_hist[lbp_hist > 0]))
    
    return {
        'LBP_Mean': lbp_mean,
        'LBP_Std': lbp_std,
        'LBP_Entropy': lbp_entropy
    }


def preprocess_image(image_path, target_size=(256, 256)):
    """Preprocessing gambar"""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        return None
    
    if target_size is not None:
        img = cv2.resize(img, (target_size[1], target_size[0]), 
                        interpolation=cv2.INTER_AREA)
    
    return img


# ============================================================================
# 2. ROUGHNESS CLASSIFICATION
# ============================================================================

def classify_roughness(glcm_features, lbp_features):
    """
    Klasifikasi tingkat kekasaran berdasarkan fitur GLCM dan LBP.
    
    Logika:
    - KASAR: Contrast tinggi, Energy rendah, LBP Entropy tinggi
    - HALUS: Contrast rendah, Energy tinggi, LBP Entropy rendah, Homogeneity tinggi
    
    Returns:
    --------
    classification : str ('kasar' atau 'halus')
    confidence : float (0.0 - 1.0)
    """
    
    # Ekstrak fitur kunci
    contrast = glcm_features['Contrast']
    homogeneity = glcm_features['Homogeneity']
    energy = glcm_features['Energy']
    lbp_entropy = lbp_features['LBP_Entropy']
    lbp_std = lbp_features['LBP_Std']
    
    # Hitung scoring untuk kasar dan halus
    # Permukaan KASAR: tinggi contrast, rendah homogeneity, tinggi LBP entropy
    roughness_score = (contrast * 0.4) + (lbp_entropy * 0.3) + (lbp_std * 0.3)
    
    # Permukaan HALUS: rendah contrast, tinggi homogeneity, rendah LBP entropy
    smoothness_score = (homogeneity * 0.4) + (energy * 0.3) + ((1 - lbp_entropy/10) * 0.3)
    
    # Normalisasi dan tentukan klasifikasi
    total_score = roughness_score + smoothness_score
    if total_score == 0:
        confidence = 0.5
        classification = 'unknown'
    else:
        roughness_prob = roughness_score / total_score
        confidence = max(roughness_prob, 1 - roughness_prob)
        classification = 'kasar' if roughness_prob > 0.5 else 'halus'
    
    return classification, confidence


def analyze_all_images(verbose=True):
    """
    Analisis semua gambar di folder data dan klasifikasikan berdasarkan kekasaran.
    
    Returns:
    --------
    results_df : DataFrame dengan hasil analisis semua gambar
    """
    
    # Cari semua gambar
    image_files = list(DATA_FOLDER.glob("*.tif")) + \
                  list(DATA_FOLDER.glob("*.jpg")) + \
                  list(DATA_FOLDER.glob("*.png"))
    
    # Filter gambar di subfolder
    image_files = [f for f in image_files 
                  if not f.parent.name.startswith('roughness_')]
    
    image_files = sorted(image_files)
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"SURFACE ROUGHNESS CLASSIFICATION SYSTEM")
        print(f"{'='*70}")
        print(f"\n[INFO] Ditemukan {len(image_files)} gambar untuk dianalisis\n")
    
    results = []
    
    # Iterate setiap gambar
    for idx, image_path in enumerate(tqdm(image_files, desc="Processing images")):
        try:
            # Preprocessing
            img_gray = preprocess_image(image_path)
            if img_gray is None:
                continue
            
            # Ekstrak fitur
            glcm_features = extract_glcm_features(img_gray)
            lbp_features = extract_lbp_features(img_gray)
            
            # Klasifikasi
            classification, confidence = classify_roughness(glcm_features, lbp_features)
            
            # Simpan hasil
            result = {
                'Filename': image_path.name,
                'FilePath': str(image_path),
                'Classification': classification,
                'Confidence': confidence,
                'GLCM_Contrast': glcm_features['Contrast'],
                'GLCM_Homogeneity': glcm_features['Homogeneity'],
                'GLCM_Energy': glcm_features['Energy'],
                'GLCM_Correlation': glcm_features['Correlation'],
                'LBP_Mean': lbp_features['LBP_Mean'],
                'LBP_Std': lbp_features['LBP_Std'],
                'LBP_Entropy': lbp_features['LBP_Entropy']
            }
            results.append(result)
        
        except Exception as e:
            if verbose:
                print(f"Error processing {image_path.name}: {str(e)}")
            continue
    
    # Convert ke DataFrame
    results_df = pd.DataFrame(results)
    
    if verbose:
        print(f"\n✓ Analisis selesai untuk {len(results)} gambar\n")
    
    return results_df


# ============================================================================
# 3. ORGANIZE & COPY IMAGES
# ============================================================================

def organize_classified_images(results_df, verbose=True):
    """
    Copy/organize gambar ke folder berdasarkan klasifikasi (kasar/halus).
    """
    
    if verbose:
        print("\n[ORGANIZING] Memisahkan gambar ke folder berdasarkan klasifikasi...\n")
    
    kasar_count = 0
    halus_count = 0
    
    for idx, row in results_df.iterrows():
        source_path = Path(row['FilePath'])
        classification = row['Classification']
        
        # Tentukan folder tujuan
        if classification == 'kasar':
            dest_folder = ROUGHNESS_KASAR_FOLDER
            kasar_count += 1
        elif classification == 'halus':
            dest_folder = ROUGHNESS_HALUS_FOLDER
            halus_count += 1
        else:
            continue
        
        # Copy gambar
        dest_path = dest_folder / source_path.name
        if not dest_path.exists():
            shutil.copy2(source_path, dest_path)
    
    if verbose:
        print(f"✓ Gambar KASAR: {kasar_count} file → {ROUGHNESS_KASAR_FOLDER.name}/")
        print(f"✓ Gambar HALUS: {halus_count} file → {ROUGHNESS_HALUS_FOLDER.name}/")
        print(f"✓ Total: {kasar_count + halus_count} file terorganisir\n")
    
    return kasar_count, halus_count


# ============================================================================
# 4. GENERATE REPORTS
# ============================================================================

def generate_reports(results_df, kasar_count, halus_count, verbose=True):
    """
    Generate laporan analisis dalam format CSV, JSON, dan TXT.
    """
    
    if verbose:
        print("[REPORTING] Membuat laporan analisis...\n")
    
    # ---- SAVE CSV ----
    results_df.to_csv(ANALYSIS_CSV, index=False)
    if verbose:
        print(f"✓ CSV Report: {ANALYSIS_CSV}")
    
    # ---- SAVE JSON ----
    results_json = {
        'metadata': {
            'total_images': len(results_df),
            'kasar_count': kasar_count,
            'halus_count': halus_count,
            'unknown_count': len(results_df) - kasar_count - halus_count
        },
        'statistics': {
            'GLCM_Contrast_Mean': float(results_df['GLCM_Contrast'].mean()),
            'GLCM_Contrast_Std': float(results_df['GLCM_Contrast'].std()),
            'GLCM_Homogeneity_Mean': float(results_df['GLCM_Homogeneity'].mean()),
            'GLCM_Energy_Mean': float(results_df['GLCM_Energy'].mean()),
            'LBP_Entropy_Mean': float(results_df['LBP_Entropy'].mean()),
            'Average_Confidence': float(results_df['Confidence'].mean())
        },
        'results': results_df.to_dict(orient='records')
    }
    
    with open(ANALYSIS_JSON, 'w') as f:
        json.dump(results_json, f, indent=2)
    if verbose:
        print(f"✓ JSON Report: {ANALYSIS_JSON}")
    
    # ---- SAVE TEXT REPORT ----
    with open(CLASSIFICATION_REPORT, 'w') as f:
        f.write("="*70 + "\n")
        f.write("SURFACE ROUGHNESS CLASSIFICATION REPORT\n")
        f.write("="*70 + "\n\n")
        
        f.write("SUMMARY\n")
        f.write("-"*70 + "\n")
        f.write(f"Total Images Analyzed:  {len(results_df)}\n")
        f.write(f"Rough (Kasar):          {kasar_count} ({100*kasar_count/len(results_df):.1f}%)\n")
        f.write(f"Smooth (Halus):         {halus_count} ({100*halus_count/len(results_df):.1f}%)\n")
        f.write(f"Unknown:                {len(results_df)-kasar_count-halus_count}\n\n")
        
        f.write("FEATURE STATISTICS\n")
        f.write("-"*70 + "\n")
        f.write(f"GLCM Contrast       - Mean: {results_df['GLCM_Contrast'].mean():.6f}, Std: {results_df['GLCM_Contrast'].std():.6f}\n")
        f.write(f"GLCM Homogeneity    - Mean: {results_df['GLCM_Homogeneity'].mean():.6f}, Std: {results_df['GLCM_Homogeneity'].std():.6f}\n")
        f.write(f"GLCM Energy         - Mean: {results_df['GLCM_Energy'].mean():.6f}, Std: {results_df['GLCM_Energy'].std():.6f}\n")
        f.write(f"LBP Entropy         - Mean: {results_df['LBP_Entropy'].mean():.6f}, Std: {results_df['LBP_Entropy'].std():.6f}\n")
        f.write(f"Avg Confidence      - {results_df['Confidence'].mean():.6f}\n\n")
        
        # High-confidence classifications
        f.write("HIGH-CONFIDENCE CLASSIFICATIONS (>80%)\n")
        f.write("-"*70 + "\n")
        high_conf = results_df[results_df['Confidence'] > 0.80]
        f.write(f"Kasar (High Conf):   {len(high_conf[high_conf['Classification']=='kasar'])} images\n")
        f.write(f"Halus (High Conf):   {len(high_conf[high_conf['Classification']=='halus'])} images\n\n")
        
        # Detailed results
        f.write("DETAILED CLASSIFICATION RESULTS\n")
        f.write("-"*70 + "\n")
        for idx, row in results_df.iterrows():
            f.write(f"{row['Filename']:<40} | {row['Classification']:<8} | Conf: {row['Confidence']:.2%}\n")
    
    if verbose:
        print(f"✓ Text Report: {CLASSIFICATION_REPORT}")
        print()


# ============================================================================
# 5. VISUALIZATION
# ============================================================================

def create_visualization(results_df, verbose=True):
    """
    Buat visualisasi hasil klasifikasi menggunakan matplotlib.
    """
    
    if verbose:
        print("[VISUALIZATION] Membuat visualisasi hasil analisis...\n")
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Surface Roughness Classification Analysis', 
                 fontsize=16, fontweight='bold')
    
    # ---- Plot 1: Distribution of Classifications ----
    classification_counts = results_df['Classification'].value_counts()
    colors_pie = ['#FF6B6B' if x == 'kasar' else '#4ECDC4' for x in classification_counts.index]
    axes[0, 0].pie(classification_counts.values, labels=classification_counts.index, 
                    autopct='%1.1f%%', colors=colors_pie, startangle=90)
    axes[0, 0].set_title('Distribution of Classifications', fontweight='bold')
    
    # ---- Plot 2: Contrast vs Homogeneity ----
    kasar = results_df[results_df['Classification'] == 'kasar']
    halus = results_df[results_df['Classification'] == 'halus']
    axes[0, 1].scatter(kasar['GLCM_Contrast'], kasar['GLCM_Homogeneity'], 
                      alpha=0.6, s=100, c='#FF6B6B', label='Kasar')
    axes[0, 1].scatter(halus['GLCM_Contrast'], halus['GLCM_Homogeneity'], 
                      alpha=0.6, s=100, c='#4ECDC4', label='Halus')
    axes[0, 1].set_xlabel('GLCM Contrast')
    axes[0, 1].set_ylabel('GLCM Homogeneity')
    axes[0, 1].set_title('Contrast vs Homogeneity', fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # ---- Plot 3: Confidence Distribution ----
    axes[0, 2].hist(results_df['Confidence'], bins=20, color='#95E1D3', edgecolor='black')
    axes[0, 2].axvline(results_df['Confidence'].mean(), color='red', 
                       linestyle='--', label=f'Mean: {results_df["Confidence"].mean():.2f}')
    axes[0, 2].set_xlabel('Confidence Score')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('Classification Confidence Distribution', fontweight='bold')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3, axis='y')
    
    # ---- Plot 4: GLCM Contrast Distribution ----
    axes[1, 0].hist(kasar['GLCM_Contrast'], bins=15, alpha=0.7, label='Kasar', color='#FF6B6B')
    axes[1, 0].hist(halus['GLCM_Contrast'], bins=15, alpha=0.7, label='Halus', color='#4ECDC4')
    axes[1, 0].set_xlabel('GLCM Contrast')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('GLCM Contrast by Classification', fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # ---- Plot 5: LBP Entropy Distribution ----
    axes[1, 1].hist(kasar['LBP_Entropy'], bins=15, alpha=0.7, label='Kasar', color='#FF6B6B')
    axes[1, 1].hist(halus['LBP_Entropy'], bins=15, alpha=0.7, label='Halus', color='#4ECDC4')
    axes[1, 1].set_xlabel('LBP Entropy')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('LBP Entropy by Classification', fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # ---- Plot 6: Feature Comparison Box Plot ----
    feature_data = [
        kasar['GLCM_Contrast'].values,
        halus['GLCM_Contrast'].values,
        kasar['LBP_Entropy'].values,
        halus['LBP_Entropy'].values
    ]
    bp = axes[1, 2].boxplot(feature_data, labels=['Kasar\nContrast', 'Halus\nContrast', 
                                                   'Kasar\nEntropy', 'Halus\nEntropy'])
    axes[1, 2].set_title('Feature Comparison', fontweight='bold')
    axes[1, 2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Simpan visualisasi
    viz_path = ANALYSIS_OUTPUT_FOLDER / "classification_visualization.png"
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    
    if verbose:
        print(f"✓ Visualization saved: {viz_path}\n")
    
    plt.show()


# ============================================================================
# 6. MAIN PIPELINE
# ============================================================================

def run_classification_pipeline(verbose=True):
    """
    Jalankan pipeline klasifikasi kekasaran permukaan lengkap.
    """
    
    try:
        # 1. Analisis semua gambar
        results_df = analyze_all_images(verbose=verbose)
        
        # 2. Organisir gambar ke folder
        kasar_count, halus_count = organize_classified_images(results_df, verbose=verbose)
        
        # 3. Generate laporan
        generate_reports(results_df, kasar_count, halus_count, verbose=verbose)
        
        # 4. Buat visualisasi
        create_visualization(results_df, verbose=verbose)
        
        # Summary
        if verbose:
            print("="*70)
            print("✓ CLASSIFICATION PIPELINE COMPLETED SUCCESSFULLY")
            print("="*70)
            print(f"\nResults Location:")
            print(f"  • Kasar images:  {ROUGHNESS_KASAR_FOLDER}")
            print(f"  • Halus images:  {ROUGHNESS_HALUS_FOLDER}")
            print(f"  • Analysis data: {ANALYSIS_OUTPUT_FOLDER}")
            print()
        
        return results_df
    
    except Exception as e:
        print(f"\n❌ Error during classification pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    """
    Main entry point untuk roughness classification system.
    """
    
    results_df = run_classification_pipeline(verbose=True)
    
    if results_df is not None:
        print("\nPress ENTER to exit...")
        input()
