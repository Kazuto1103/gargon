"""Dataset Integrity Verification Module.

This script validates the generated synthetic_dataset.json to ensure it meets
all schema, uniqueness, class balance, and BIO-tag transition constraints.
"""

import json
import os
import sys
from typing import Any, Dict, List


def verify_dataset(file_path: str) -> None:
    """Verifies the integrity of the generated dataset.

    Args:
        file_path (str): Path to the dataset JSON file.

    Raises:
        ValueError: If any validation rule is violated.
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        dataset: List[Dict[str, Any]] = json.load(f)

    # 1. Total Size check
    total_rows = len(dataset)
    print(f"Total rows found: {total_rows}")
    if total_rows < 2500:
        raise ValueError(f"Dataset too small: {total_rows} rows (expected >= 2500)")

    seen_sentences = set()
    intent_counts: Dict[str, int] = {}
    slot_distribution: Dict[str, int] = {}

    for index, item in enumerate(dataset):
        # 2. Schema check
        for field in ["sentence", "intent", "tokens", "slots"]:
            if field not in item:
                raise ValueError(f"Item at index {index} is missing field '{field}'")

        sentence = item["sentence"]
        intent = item["intent"]
        tokens = item["tokens"]
        slots = item["slots"]

        # 3. Uniqueness check
        if sentence in seen_sentences:
            raise ValueError(f"Duplicate sentence detected at index {index}: '{sentence}'")
        seen_sentences.add(sentence)

        # 4. Token & Slot alignment check
        if len(tokens) != len(slots):
            raise ValueError(
                f"Token and slot mismatch at index {index}: "
                f"{len(tokens)} tokens vs {len(slots)} slots. Sentence: '{sentence}'"
            )

        # Count intent
        intent_counts[intent] = intent_counts.get(intent, 0) + 1

        # 5. BIO Tag validation
        for i, tag in enumerate(slots):
            slot_distribution[tag] = slot_distribution.get(tag, 0) + 1
            if tag == "O":
                continue

            # Ensure valid format (starts with B- or I-)
            if not (tag.startswith("B-") or tag.startswith("I-")):
                raise ValueError(f"Invalid tag format '{tag}' at index {index}, token position {i}")

            # Check transitions: I- must be preceded by B- or I- of the same type
            if tag.startswith("I-"):
                slot_type = tag[2:]
                if i == 0:
                    raise ValueError(
                        f"Sentence cannot start with an I- tag '{tag}' at index {index}. "
                        f"Sentence: '{sentence}'"
                    )
                prev_tag = slots[i - 1]
                if prev_tag not in (f"B-{slot_type}", f"I-{slot_type}"):
                    raise ValueError(
                        f"Invalid BIO sequence at index {index}: '{prev_tag}' followed by '{tag}'. "
                        f"Sentence: '{sentence}'"
                    )

    # 6. Intent Balance check
    print("\n--- Intent Distribution ---")
    for intent, count in intent_counts.items():
        print(f" - {intent}: {count}")
        if count < 500:
            raise ValueError(f"Intent '{intent}' has only {count} rows (expected >= 500)")

    # Print Slot Statistics
    print("\n--- Slot Distribution ---")
    for tag, count in sorted(slot_distribution.items()):
        print(f" - {tag}: {count}")

    print("\nSUCCESS: All dataset integrity checks PASSED successfully!")


def main() -> None:
    """Main execution block to run the dataset verification."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(current_dir, "synthetic_dataset.json")

    print(f"Starting verification of dataset at: {dataset_path}")
    try:
        verify_dataset(dataset_path)
    except Exception as e:
        print(f"Verification FAILED: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
