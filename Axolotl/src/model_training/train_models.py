"""Model Training and Evaluation Module.

This script trains a LinearSVC model for Intent Classification (with Stratified
Cross-Validation) and a CRF model for Slot Filling (using a custom picklable wrapper
for pycrfsuite). It evaluates both models and serializes them to the build directory.
All output is captured in a log file.
"""

import os
import pickle
import sys
import tempfile
from typing import Any, Dict, List, Tuple
import pycrfsuite
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.svm import LinearSVC


class DualWriter:
    """Tee writer that redirects stdout to both terminal and a log file."""

    def __init__(self, filepath: str):
        """Initializes the DualWriter.

        Args:
            filepath (str): Path to the log file.
        """
        self.terminal = sys.stdout
        self.log = open(filepath, "w", encoding="utf-8")

    def write(self, message: str) -> None:
        """Writes message to terminal and log file.

        Args:
            message (str): String content to write.
        """
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self) -> None:
        """Flushes the streams."""
        self.terminal.flush()
        self.log.flush()


class CRFModel:
    """A picklable wrapper around python-crfsuite Trainer and Tagger."""

    def __init__(self):
        """Initializes the CRFModel."""
        self.model_data: bytes = b""
        self._tagger: pycrfsuite.Tagger = None

    def fit(self, X: List[List[Dict[str, Any]]], y: List[List[str]], c1: float = 0.1, c2: float = 0.1, max_iterations: int = 100) -> None:
        """Trains the CRF model and stores binary weights in RAM.

        Args:
            X (List[List[Dict[str, Any]]]): Sequence features.
            y (List[List[str]]): Sequence labels (BIO tags).
            c1 (float): L1 regularization parameter.
            c2 (float): L2 regularization parameter.
            max_iterations (int): Maximum L-BFGS iterations.
        """
        temp_fd, temp_path = tempfile.mkstemp()
        try:
            trainer = pycrfsuite.Trainer(verbose=False)
            for xseq, yseq in zip(X, y):
                trainer.append(xseq, yseq)

            trainer.set_params({
                "c1": c1,
                "c2": c2,
                "max_iterations": max_iterations,
                "feature.possible_transitions": True
            })
            trainer.train(temp_path)

            with open(temp_path, "rb") as f:
                self.model_data = f.read()
        finally:
            os.close(temp_fd)
            if os.path.exists(temp_path):
                os.remove(temp_path)

        self._init_tagger()

    def _init_tagger(self) -> None:
        """Lazily instantiates the underlying pycrfsuite.Tagger from bytes."""
        if not self.model_data:
            return
        self._tagger = pycrfsuite.Tagger()
        temp_fd, temp_path = tempfile.mkstemp()
        try:
            with open(temp_path, "wb") as f:
                f.write(self.model_data)
            self._tagger.open(temp_path)
        finally:
            os.close(temp_fd)
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def predict(self, X: List[List[Dict[str, Any]]]) -> List[List[str]]:
        """Predicts the sequence tags for a list of sentence sequences.

        Args:
            X (List[List[Dict[str, Any]]]): Sequence features.

        Returns:
            List[List[str]]: Predicted sequence labels.
        """
        if self._tagger is None:
            self._init_tagger()
        return [self._tagger.tag(xseq) for xseq in X]

    def __getstate__(self) -> Dict[str, Any]:
        """Custom pickling protocol to save model bytes.

        Returns:
            Dict[str, Any]: Pickle state dictionary.
        """
        return {"model_data": self.model_data}

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Custom unpickling protocol.

        Args:
            state (Dict[str, Any]): Pickled state dictionary.
        """
        self.model_data = state["model_data"]
        self._tagger = None  # Re-initialized lazily upon predict invocation


def main() -> None:
    """Main execution block to set up logging, train, evaluate, and save models."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(current_dir, "log")
    build_dir = os.path.join(current_dir, "build")

    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "train_models.log")

    # Redirect stdout and stderr to both file and terminal
    sys.stdout = DualWriter(log_path)
    sys.stderr = sys.stdout

    print("==================================================")
    print("        STARTING MODEL TRAINING PIPELINE          ")
    print("==================================================")

    # 1. Loading Preprocessed Datasets
    svm_data_path = os.path.join(build_dir, "svm_data.pkl")
    crf_data_path = os.path.join(build_dir, "crf_data.pkl")

    print(f"Loading SVM data from: {svm_data_path}")
    with open(svm_data_path, "rb") as f:
        svm_data = pickle.load(f)

    print(f"Loading CRF data from: {crf_data_path}")
    with open(crf_data_path, "rb") as f:
        crf_data = pickle.load(f)

    X_train_svm = svm_data["X_train"]
    X_test_svm = svm_data["X_test"]
    y_train_svm = svm_data["y_train"]
    y_test_svm = svm_data["y_test"]

    X_train_crf = crf_data["X_train"]
    X_test_crf = crf_data["X_test"]
    y_train_crf = crf_data["y_train"]
    y_test_crf = crf_data["y_test"]

    # 2. SVM Intent Classifier Training
    print("\n--- Training SVM Intent Classifier ---")
    # Setting conservative C=0.5 to prevent overfitting to synthetic templates
    svm_model = LinearSVC(C=0.5, random_state=42)

    # 5-Fold Stratified Cross-Validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(svm_model, X_train_svm, y_train_svm, cv=cv)
    print(f"5-Fold Cross-Validation Scores: {cv_scores}")
    print(f"Mean CV Accuracy: {cv_scores.mean():.4f} (std: {cv_scores.std():.4f})")

    # Fit final model on the full training set
    print("Fitting final SVM model on 100% of training data...")
    svm_model.fit(X_train_svm, y_train_svm)

    # 3. CRF Slot Filler Training
    print("\n--- Training CRF Slot Filler ---")
    crf_model = CRFModel()
    print("Fitting CRF with L1=0.1, L2=0.1, max_iterations=100...")
    crf_model.fit(X_train_crf, y_train_crf, c1=0.1, c2=0.1, max_iterations=100)

    # 4. Evaluation
    print("\n==================================================")
    print("               EVALUATION REPORTS                 ")
    print("==================================================")

    # SVM Evaluation
    print("\n[SVM Intent Classifier Report]")
    y_pred_svm = svm_model.predict(X_test_svm)
    print(classification_report(y_test_svm, y_pred_svm))

    # CRF Evaluation
    print("\n[CRF Slot Filler Report (BIO Entity Tags, excluding 'O')]")
    y_pred_crf = crf_model.predict(X_test_crf)

    # Flatten lists for scikit-learn metric reporting
    y_true_flat = [tag for seq in y_test_crf for tag in seq]
    y_pred_flat = [tag for seq in y_pred_crf for tag in seq]

    # Exclude 'O' to focus on slot extraction metrics
    unique_tags = sorted(list(set(y_true_flat) - {"O"}))
    print(classification_report(y_true_flat, y_pred_flat, labels=unique_tags))

    # 5. Serialization
    print("\n==================================================")
    print("               SERIALIZING MODELS                 ")
    print("==================================================")
    intent_model_path = os.path.join(build_dir, "intent_model.pkl")
    slot_model_path = os.path.join(build_dir, "slot_model.pkl")

    with open(intent_model_path, "wb") as f:
        pickle.dump(svm_model, f)
    print(f"Successfully saved SVM Intent Classifier -> {intent_model_path}")

    with open(slot_model_path, "wb") as f:
        pickle.dump(crf_model, f)
    print(f"Successfully saved CRF Slot Filler        -> {slot_model_path}")

    print("\nTraining and Evaluation pipeline complete!")


if __name__ == "__main__":
    main()
