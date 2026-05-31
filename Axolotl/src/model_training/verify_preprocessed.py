"""Preprocessing Verification Module.

This script validates the serialized pickle outputs in the 'build/' directory,
ensuring shapes, split ratios, types, and alignments are correct.
"""

import os
import pickle
import sys
from typing import Any, Dict


def verify_pickles(build_dir: str) -> None:
    """Loads and validates the properties of each pickle artifact.

    Args:
        build_dir (str): Path to the build output directory.

    Raises:
        ValueError: If any validation rule is violated.
        FileNotFoundError: If any expected file is missing.
    """
    vectorizer_path = os.path.join(build_dir, "vectorizer.pkl")
    svm_path = os.path.join(build_dir, "svm_data.pkl")
    crf_path = os.path.join(build_dir, "crf_data.pkl")

    # 1. Existence Check
    for path in [vectorizer_path, svm_path, crf_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing build artifact: {path}")

    # 2. Vectorizer Check
    print("Verifying vectorizer.pkl...")
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
    if not hasattr(vectorizer, "vocabulary_"):
        raise ValueError("Vectorizer is not fitted (missing vocabulary_ attribute)")
    vocab_size = len(vectorizer.vocabulary_)
    print(f" - Vectorizer fitted. Vocabulary size: {vocab_size}")

    # 3. SVM Data Check
    print("Verifying svm_data.pkl...")
    with open(svm_path, "rb") as f:
        svm_data: Dict[str, Any] = pickle.load(f)

    for key in ["X_train_raw", "X_test_raw", "X_train", "X_test", "y_train", "y_test"]:
        if key not in svm_data:
            raise ValueError(f"svm_data is missing key: '{key}'")

    train_size = len(svm_data["y_train"])
    test_size = len(svm_data["y_test"])
    print(f" - SVM Train size: {train_size}")
    print(f" - SVM Test size: {test_size}")

    if train_size != 2200 or test_size != 550:
        raise ValueError(f"SVM Split sizes incorrect (expected 2200/550, got {train_size}/{test_size})")

    # Matrix shapes check
    train_shape = svm_data["X_train"].shape
    test_shape = svm_data["X_test"].shape
    print(f" - SVM X_train matrix shape: {train_shape}")
    print(f" - SVM X_test matrix shape: {test_shape}")

    if train_shape[0] != 2200 or train_shape[1] != vocab_size:
        raise ValueError(f"SVM X_train shape mismatch: {train_shape}")
    if test_shape[0] != 550 or test_shape[1] != vocab_size:
        raise ValueError(f"SVM X_test shape mismatch: {test_shape}")

    # 4. CRF Data Check
    print("Verifying crf_data.pkl...")
    with open(crf_path, "rb") as f:
        crf_data: Dict[str, Any] = pickle.load(f)

    for key in ["X_train", "X_test", "y_train", "y_test"]:
        if key not in crf_data:
            raise ValueError(f"crf_data is missing key: '{key}'")

    crf_train_size = len(crf_data["X_train"])
    crf_test_size = len(crf_data["X_test"])
    print(f" - CRF Train size: {crf_train_size}")
    print(f" - CRF Test size: {crf_test_size}")

    if crf_train_size != 2200 or crf_test_size != 550:
        raise ValueError(f"CRF Split sizes incorrect (expected 2200/550, got {crf_train_size}/{crf_test_size})")

    # Alignments check
    for i in range(crf_train_size):
        x_len = len(crf_data["X_train"][i])
        y_len = len(crf_data["y_train"][i])
        if x_len != y_len:
            raise ValueError(f"CRF Train token-feature alignment mismatch at index {i}: {x_len} features vs {y_len} tags")

    for i in range(crf_test_size):
        x_len = len(crf_data["X_test"][i])
        y_len = len(crf_data["y_test"][i])
        if x_len != y_len:
            raise ValueError(f"CRF Test token-feature alignment mismatch at index {i}: {x_len} features vs {y_len} tags")

    print("\nSUCCESS: All serialized features and splits are completely INTEGRAL and VALID!")


def main() -> None:
    """Main execution block to run pickle checks."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    build_path = os.path.join(current_dir, "build")

    print(f"Starting verification of build files in: {build_path}")
    try:
        verify_pickles(build_path)
    except Exception as e:
        print(f"Verification FAILED: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
