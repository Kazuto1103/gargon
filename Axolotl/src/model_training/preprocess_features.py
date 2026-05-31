"""Data Preprocessing and Feature Extraction Module.

This script loads the synthetic dataset, performs sentence-level preprocessing
and TF-IDF vectorization for SVM, extracts token-level sequence features for CRF,
splits the data, and serializes the processed outputs.
"""

import json
import os
import pickle
import re
from typing import Any, Dict, List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from normalizer import normalize_text, tokenize_sentence


def preprocess_text(text: str) -> str:
    """Preprocesses and normalizes raw sentence text.

    Applies canonical normalization (abbreviation expansion, number-unit spacing)
    via normalizer.py, then performs standard lowercase and punctuation cleanup.

    Args:
        text (str): The raw input sentence.

    Returns:
        str: The fully normalized and preprocessed sentence.
    """
    # Apply canonical normalization first (handles typos, abbreviations, merged words)
    text = normalize_text(text)
    # Remove residual punctuation and normalize whitespace
    text = re.sub(r"[.,\/#!$%\^&\*;:{}=\-_`~()?]", " ", text)
    return " ".join(text.split())


def word2features(sent: List[str], i: int) -> Dict[str, Any]:
    """Extracts features for a single word at index i in a token sequence.

    Args:
        sent (List[str]): The sequence of tokens (sentence).
        i (int): The index of the target word.

    Returns:
        Dict[str, Any]: A dictionary of linguistic features for the word.
    """
    word = sent[i]
    features = {
        "bias": 1.0,
        "word.lower()": word.lower(),
        "word[-3:]": word[-3:] if len(word) >= 3 else word,
        "word[-2:]": word[-2:] if len(word) >= 2 else word,
        "word.isdigit()": word.isdigit(),
    }

    # Contextual window: Previous word (i-1)
    if i > 0:
        word_prev = sent[i - 1]
        features.update({
            "-1:word.lower()": word_prev.lower(),
            "-1:word.isdigit()": word_prev.isdigit(),
        })
    else:
        features["BOS"] = True

    # Contextual window: Next word (i+1)
    if i < len(sent) - 1:
        word_next = sent[i + 1]
        features.update({
            "+1:word.lower()": word_next.lower(),
            "+1:word.isdigit()": word_next.isdigit(),
        })
    else:
        features["EOS"] = True

    return features


def sent2features(sent: List[str]) -> List[Dict[str, Any]]:
    """Converts a token sequence into a sequence of feature dictionaries.

    Args:
        sent (List[str]): The token sequence.

    Returns:
        List[Dict[str, Any]]: List of feature dictionaries.
    """
    return [word2features(sent, i) for i in range(len(sent))]


def main() -> None:
    """Main execution block to load data, build features, split, and serialize."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(current_dir, "synthetic_dataset.json")
    build_dir = os.path.join(current_dir, "build")

    print(f"Loading dataset from: {dataset_path}")
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    print("Splitting dataset into 80% train and 20% test sets (stratified by intent)...")
    # Extract intents for stratification to ensure train/test sets are perfectly balanced
    intents = [item["intent"] for item in dataset]
    train_data, test_data = train_test_split(
        dataset, test_size=0.2, random_state=42, stratify=intents
    )

    # 1. SVM Feature Pipeline
    print("Building SVM features...")
    X_train_svm_raw = [preprocess_text(item["sentence"]) for item in train_data]
    X_test_svm_raw = [preprocess_text(item["sentence"]) for item in test_data]
    y_train_svm = [item["intent"] for item in train_data]
    y_test_svm = [item["intent"] for item in test_data]

    # Fit TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    X_train_svm = vectorizer.fit_transform(X_train_svm_raw)
    X_test_svm = vectorizer.transform(X_test_svm_raw)

    # 2. CRF Feature Pipeline
    # Use original tokens from dataset for training: normalization may change token count
    # which would break BIO slot alignment. Normalization is applied only at INFERENCE time
    # for new user inputs (in stress_test and verify_models).
    print("Building CRF features...")
    X_train_crf = [sent2features(item["tokens"]) for item in train_data]
    X_test_crf = [sent2features(item["tokens"]) for item in test_data]
    y_train_crf = [item["slots"] for item in train_data]
    y_test_crf = [item["slots"] for item in test_data]

    # Ensure build directory exists
    os.makedirs(build_dir, exist_ok=True)

    # 3. Serialization
    print("Serializing vectorizer and datasets...")
    vectorizer_path = os.path.join(build_dir, "vectorizer.pkl")
    svm_data_path = os.path.join(build_dir, "svm_data.pkl")
    crf_data_path = os.path.join(build_dir, "crf_data.pkl")

    with open(vectorizer_path, "wb") as f:
        pickle.dump(vectorizer, f)

    svm_payload = {
        "X_train_raw": X_train_svm_raw,
        "X_test_raw": X_test_svm_raw,
        "X_train": X_train_svm,
        "X_test": X_test_svm,
        "y_train": y_train_svm,
        "y_test": y_test_svm
    }
    with open(svm_data_path, "wb") as f:
        pickle.dump(svm_payload, f)

    crf_payload = {
        "X_train": X_train_crf,
        "X_test": X_test_crf,
        "y_train": y_train_crf,
        "y_test": y_test_crf
    }
    with open(crf_data_path, "wb") as f:
        pickle.dump(crf_payload, f)

    print("Success: All components have been serialized to the 'build/' directory!")
    print(f" - Vectorizer: {vectorizer_path}")
    print(f" - SVM Data: {svm_data_path}")
    print(f" - CRF Data: {crf_data_path}")


if __name__ == "__main__":
    main()
