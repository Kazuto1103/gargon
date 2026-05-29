"""
Quick Texture Analysis - Analisis Cepat 2 Gambar Permukaan
Tanpa visualisasi matplotlib interactive

Author: Nouzen
Date: 2026
"""

import cv2
import numpy as np
from pathlib import Path
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage import img_as_ubyte
import warnings
warnings.filterwarnings('ignore')

# Import functions dari texture_analyzer
import sys
sys.path.insert(0, str(Path(__file__).parent))
from texture_analyzer import (extract_glcm_features, extract_lbp_features, 
                              preprocess_image, print_analysis_results)

def quick_analysis():
    """Analisis cepat dua gambar kasar"""
    
    # Path folders
    data_folder = Path(__file__).parent.parent / "data"
    kasar_folder = data_folder / "roughness_kasar"
    
    # Cari gambar di folder kasar
    image_files = sorted(list(kasar_folder.glob("*.tif")) + 
                        list(kasar_folder.glob("*.tiff")))
    
    if len(image_files) < 2:
        print("❌ Minimal 2 gambar kasar diperlukan!")
        return
    
    # Ambil 2 gambar pertama
    image_path_1 = image_files[0]
    image_path_2 = image_files[1]
    
    print("\n" + "="*70)
    print("SURFACE ROUGHNESS TEXTURE ANALYSIS - GLCM vs LBP COMPARISON")
    print("="*70 + "\n")
    
    print("[1/3] Preprocessing gambar...")
    img_gray_1, img_rgb_1 = preprocess_image(image_path_1, target_size=(256, 256))
    img_gray_2, img_rgb_2 = preprocess_image(image_path_2, target_size=(256, 256))
    
    print("\n[2/3] Ekstraksi fitur GLCM dan LBP...")
    glcm_features_1, _ = extract_glcm_features(img_gray_1)
    glcm_features_2, _ = extract_glcm_features(img_gray_2)
    
    lbp_hist_1, lbp_map_1, lbp_mean_1, lbp_std_1 = extract_lbp_features(img_gray_1)
    lbp_hist_2, lbp_map_2, lbp_mean_2, lbp_std_2 = extract_lbp_features(img_gray_2)
    
    print("✓ GLCM features extracted from both images")
    print("✓ LBP features extracted from both images")
    
    print("\n[3/3] Hasil Analisis...")
    print_analysis_results(
        glcm_features_1, glcm_features_2,
        lbp_mean_1, lbp_mean_2,
        lbp_std_1, lbp_std_2,
        (image_files[0].stem, image_files[1].stem)
    )

if __name__ == "__main__":
    quick_analysis()
