"""End-to-End Robotics NLU Pipeline and Simulation.

This module integrates all phases of the "Grounded NLP: Text-to-Action Sequence
Generator" project:
1. Text Normalization (normalizer.py)
2. Machine Learning Inference (SVM Intent Classification & CRF Slot Filling)
3. Deterministic Grounding Translation (grounding_translator.py)

It provides a unified entry point class `NLUPipeline`, an automated simulation
test suite, and an interactive CLI REPL for real-time testing.
"""

import json
import os
import pickle
import sys
from typing import Any, Dict, List, Tuple

# Add src directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import pipeline components
from core_pipeline.normalizer import normalize_text, tokenize_sentence
from model_training.preprocess_features import preprocess_text, sent2features
from model_training.train_models import CRFModel
from core_pipeline.grounding_translator import GroundingEngine


class NLUPipeline:
    """Unified entry point for the Grounded NLU pipeline."""

    def __init__(self, build_dir: str = "build") -> None:
        """Initializes the pipeline by loading all serialized ML models.

        Args:
            build_dir (str): Directory containing the serialized model binaries.

        Raises:
            FileNotFoundError: If any of the model files are missing.
        """
        self.build_dir = build_dir
        self.vectorizer_path = os.path.join(build_dir, "vectorizer.pkl")
        self.intent_model_path = os.path.join(build_dir, "intent_model.pkl")
        self.slot_model_path = os.path.join(build_dir, "slot_model.pkl")

        # Load vectorizer
        if not os.path.exists(self.vectorizer_path):
            raise FileNotFoundError(f"Vectorizer not found at: {self.vectorizer_path}")
        with open(self.vectorizer_path, "rb") as f:
            self.vectorizer = pickle.load(f)

        # Load SVM Intent Classifier
        if not os.path.exists(self.intent_model_path):
            raise FileNotFoundError(f"Intent model not found at: {self.intent_model_path}")
        with open(self.intent_model_path, "rb") as f:
            self.svm_model = pickle.load(f)

        # Load CRF Slot Filler
        if not os.path.exists(self.slot_model_path):
            raise FileNotFoundError(f"Slot model not found at: {self.slot_model_path}")
        with open(self.slot_model_path, "rb") as f:
            self.crf_model = pickle.load(f)

        # Initialize Grounding Engine
        self.grounding_engine = GroundingEngine()

    def predict_and_ground(self, raw_text: str) -> Dict[str, Any]:
        """Runs a raw command string through the entire NLU grounding pipeline.

        Steps:
        1. Normalize raw Indonesian text (expands abbreviations, fixes typos/spaces).
        2. Preprocess text for SVM and transform via TF-IDF vectorizer.
        3. Predict command intent using SVM model.
        4. Tokenize the normalized text and extract CRF-specific features.
        5. Predict BIO tag slots using CRF model.
        6. Translate intent and (token, slot) pairs into a canonical grounded JSON.

        Args:
            raw_text (str): The raw input query from the user.

        Returns:
            Dict[str, Any]: The canonical grounded JSON command payload.
        """
        # Step 1: Normalize text
        normalized_text = normalize_text(raw_text)

        # Step 2: SVM Preprocessing and Inference
        clean_text_svm = preprocess_text(raw_text)
        vectorized = self.vectorizer.transform([clean_text_svm])
        predicted_intent = self.svm_model.predict(vectorized)[0]

        # Step 3: CRF Tokenization and Inference
        tokens = tokenize_sentence(normalized_text)
        features = sent2features(tokens)
        predicted_slots = self.crf_model.predict([features])[0]

        # Combine tokens and predicted slot tags
        token_tag_pairs = list(zip(tokens, predicted_slots))

        # Step 4: Grounding Translation
        grounded_payload = self.grounding_engine.translate(
            intent=predicted_intent,
            token_tag_pairs=token_tag_pairs
        )

        return grounded_payload


# ---------------------------------------------------------------------------
# Output Redirection Tee Class
# ---------------------------------------------------------------------------

class TeeWriter:
    """Tee utility to write output to both stdout and a log file simultaneously."""

    def __init__(self, log_path: str) -> None:
        """Initializes the TeeWriter.

        Args:
            log_path (str): File path where logs should be saved.
        """
        self.file = open(log_path, "w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, data: str) -> None:
        """Writes data to both console and file.

        Args:
            data (str): The string to write.
        """
        self.stdout.write(data)
        if not self.file.closed:
            self.file.write(data)
            self.file.flush()

    def flush(self) -> None:
        """Flushes both stdout and file streams."""
        self.stdout.flush()
        if not self.file.closed:
            self.file.flush()

    def close(self) -> None:
        """Closes the log file."""
        if not self.file.closed:
            self.file.close()
        if sys.stdout is self:
            sys.stdout = self.stdout


# ---------------------------------------------------------------------------
# Execution and Verification Harness
# ---------------------------------------------------------------------------

def run_automated_tests(pipeline: NLUPipeline) -> None:
    """Executes a diverse set of test cases to verify the NLU integration.

    Args:
        pipeline (NLUPipeline): The initialized pipeline instance.
    """
    print("=" * 60)
    print("       STARTING AUTOMATED PIPELINE INTEGRATION TESTS        ")
    print("=" * 60)

    # 12 representative test cases covering all intents, fallbacks, and noise
    test_cases = [
        # 1. Direct command (normal)
        "maju ke depan dua meter",
        # 2. Direct command with abbreviations and typo merges
        "mju dpan 2 mtr",
        # 3. Scheduled command (normal)
        "putar ke kiri 90 derajat besok jam 3 sore",
        # 4. Scheduled command with abbreviations
        "geser kekiri dua meter bsk pagi",
        # 5. Stop command (normal)
        "berhenti sekarang juga",
        # 6. Stop command (slang & noise bypass)
        "woi robot berhenti secepatnya dong",
        # 7. Repeated command
        "tiap 5 mnt belok kekanan 90 deg",
        # 8. Repeated command (synonym + textual times)
        "setiap jam muter balik dua kali",
        # 9. Missing Direction (direct command fallback)
        "maju 5 meter",
        # 10. CRF Slot Anomaly (non-numeric value in Quantity)
        "angkat naik meter",  # 'naik' might be misidentified, testing fallback
        # 11. Invalid/Out-of-Domain text
        "hari ini cuaca cerah banget ya robot",
        # 12. Normalizer spacing fix test
        "mundur 10cm skrg"
    ]

    for i, raw_sent in enumerate(test_cases, 1):
        print(f"\n[TEST CASE #{i}] Input: \"{raw_sent}\"")
        try:
            payload = pipeline.predict_and_ground(raw_sent)
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        except Exception as e:
            print(f"ERROR executing test case: {e}", file=sys.stderr)
        print("-" * 60)

    print("\nAutomated integration tests completed successfully.")


def run_interactive_repl(pipeline: NLUPipeline) -> None:
    """Enters an interactive console loop for manual real-time command testing.

    Args:
        pipeline (NLUPipeline): The initialized pipeline instance.
    """
    print("=" * 60)
    print("          NLU PIPELINE INTERACTIVE SIMULATOR (CLI)          ")
    print("=" * 60)
    print("Type your Indonesian robotic command below and press Enter.")
    print("Type 'exit', 'quit', or 'q' to end the session.\n")

    while True:
        try:
            raw_input = input("robot-nlu> ").strip()
            if not raw_input:
                continue
            if raw_input.lower() in ("exit", "quit", "q"):
                print("Exiting simulator. Goodbye!")
                break

            payload = pipeline.predict_and_ground(raw_input)
            print("\nCanonical JSON Payload:")
            print(json.dumps(payload, indent=2, ensure_ascii=False))
            print("-" * 40)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting simulator. Goodbye!")
            break
        except Exception as e:
            print(f"Error: {e}\n")


def main() -> None:
    """Main execution point for Phase 6 E2E Integration and Simulation."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_root = os.path.dirname(os.path.dirname(current_dir))  # Navigate to workspace root
    log_dir = os.path.join(current_dir, "log")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "e2e_simulation.log")

    # Set up dual output to log file and console
    tee = TeeWriter(log_path)
    sys.stdout = tee

    try:
        # Initialize pipeline with models from workspace root
        build_dir = os.path.join(workspace_root, "build")
        pipeline = NLUPipeline(build_dir=build_dir)

        # Check for interactive flag
        if "--interactive" in sys.argv or "-i" in sys.argv:
            # For interactive mode, we disable teeing stdout to avoid duplicate prints during inputs
            sys.stdout = tee.stdout
            tee.close()
            run_interactive_repl(pipeline)
        else:
            run_automated_tests(pipeline)
            print(f"\nFull execution log successfully written to: {log_path}")
            tee.close()

    except Exception as e:
        print(f"Initialization or Execution Failed: {e}", file=sys.stderr)
        if hasattr(tee, "close"):
            tee.close()
        sys.exit(1)


if __name__ == "__main__":
    main()
