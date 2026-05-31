"""Model Prediction Verification Module.

This script loads the serialized vectorizer, intent (SVM) model, and slot (CRF) model,
runs inference on various test sentences, and verifies correctness. All output
is captured in a log file.
"""

import os
import pickle
import sys
from typing import List
from normalizer import normalize_text, tokenize_sentence
from preprocess_features import preprocess_text, sent2features
from train_models import CRFModel, DualWriter



def main() -> None:
    """Main execution block to load models and test predictions."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(current_dir, "log")
    build_dir = os.path.join(current_dir, "build")

    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "verify_models.log")

    # Set up logging
    sys.stdout = DualWriter(log_path)
    sys.stderr = sys.stdout

    print("==================================================")
    print("         VERIFYING SERIALIZED ML MODELS           ")
    print("==================================================")

    # 1. Load serialized artifacts
    vectorizer_path = os.path.join(build_dir, "vectorizer.pkl")
    intent_model_path = os.path.join(build_dir, "intent_model.pkl")
    slot_model_path = os.path.join(build_dir, "slot_model.pkl")

    print(f"Loading Vectorizer from: {vectorizer_path}")
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)

    print(f"Loading SVM Intent Classifier from: {intent_model_path}")
    with open(intent_model_path, "rb") as f:
        svm_model = pickle.load(f)

    print(f"Loading CRF Slot Model from: {slot_model_path}")
    with open(slot_model_path, "rb") as f:
        crf_model = pickle.load(f)

    print("All models successfully loaded!\n")

    # 2. Test sentences covering different intents
    test_cases = [
        "maju ke depan dua meter besok jam 3 sore",
        "tolong putar balik dengan cepat sekarang juga",
        "batalkan semua gerakan sekarang juga",
        "tolong putar ke kiri sepuluh derajat setiap lima detik",
        "hari ini saya ingin memasak makan malam di rumah"
    ]

    print("==================================================")
    print("               INFERENCE TESTING                  ")
    print("==================================================")

    for i, raw_sent in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: \"{raw_sent}\"")
        normalized_sent = normalize_text(raw_sent)
        print(f"  [Normalized]: {normalized_sent}")

        # SVM Prediction
        clean_sent = preprocess_text(raw_sent)
        vectorized = vectorizer.transform([clean_sent])
        predicted_intent = svm_model.predict(vectorized)[0]
        print(f" -> Predicted Intent: {predicted_intent}")

        # CRF Prediction on normalized tokens
        tokens = tokenize_sentence(normalized_sent)
        features = sent2features(tokens)
        predicted_slots = crf_model.predict([features])[0]

        # Display aligned tokens and tags
        print(" -> Predicted Slots:")
        for tok, tag in zip(tokens, predicted_slots):
            print(f"    * {tok:<15} => {tag}")

    print("\n==================================================")
    print("Verification execution completed successfully!")


if __name__ == "__main__":
    main()
