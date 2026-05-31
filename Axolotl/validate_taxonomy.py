"""Taxonomy Validation Module.

This script validates the structure, types, and consistency of the taxonomy.json file,
ensuring it serves as a reliable Single Source of Truth for the NLU components.
"""

import json
import os
import sys
from typing import Any, Dict


def load_taxonomy(file_path: str) -> Dict[str, Any]:
    """Loads the taxonomy JSON file from the specified path.

    Args:
        file_path (str): The absolute or relative path to the taxonomy.json file.

    Returns:
        Dict[str, Any]: The parsed JSON content of the taxonomy.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_structure(taxonomy: Dict[str, Any]) -> None:
    """Validates the basic structure and presence of mandatory keys in the taxonomy.

    Args:
        taxonomy (Dict[str, Any]): The loaded taxonomy dictionary.

    Raises:
        ValueError: If any mandatory key or structure is invalid.
    """
    required_keys = ["project_name", "version", "intents", "slots", "vocabulary_boundary", "grounding_rules"]
    for key in required_keys:
        if key not in taxonomy:
            raise ValueError(f"Missing required top-level key: '{key}'")

    # Validate Intents
    for intent_name, intent_data in taxonomy["intents"].items():
        if "execution_type" not in intent_data:
            raise ValueError(f"Intent '{intent_name}' is missing 'execution_type'")
        if "description" not in intent_data:
            raise ValueError(f"Intent '{intent_name}' is missing 'description'")
        if "examples" not in intent_data or not isinstance(intent_data["examples"], list):
            raise ValueError(f"Intent '{intent_name}' must have a list of 'examples'")

    # Validate Slots
    for slot_name, slot_data in taxonomy["slots"].items():
        if "tag_bio" not in slot_data or not isinstance(slot_data["tag_bio"], list):
            raise ValueError(f"Slot '{slot_name}' must have a list of 'tag_bio'")


def validate_vocabulary_and_grounding(taxonomy: Dict[str, Any]) -> None:
    """Verifies that the grounding rules correspond directly to vocabulary boundary terms.

    Args:
        taxonomy (Dict[str, Any]): The loaded taxonomy dictionary.

    Raises:
        ValueError: If grounding mappings use words that are not defined in vocabulary boundaries.
    """
    vocab = taxonomy["vocabulary_boundary"]
    rules = taxonomy["grounding_rules"]

    # Pair mappings to vocabs for cross validation
    mapping_checks = [
        ("action_mapping", "ACTION"),
        ("direction_mapping", "DIRECTION"),
        ("modifier_mapping", "MODIFIER"),
        ("unit_mapping", "UNIT"),
        ("quantity_word_mapping", "QUANTITY_WORDS"),
    ]

    for rule_key, vocab_key in mapping_checks:
        if rule_key not in rules:
            raise ValueError(f"Missing '{rule_key}' in grounding_rules")
        if vocab_key not in vocab:
            raise ValueError(f"Missing '{vocab_key}' in vocabulary_boundary")

        rule_items = rules[rule_key]
        vocab_set = set(vocab[vocab_key])

        for word in rule_items.keys():
            if word not in vocab_set:
                raise ValueError(
                    f"Word '{word}' in grounding rule '{rule_key}' "
                    f"is not defined in vocabulary boundary '{vocab_key}'"
                )

    print("SUCCESS: Vocabulary and grounding rules are fully aligned!")


def main() -> None:
    """Main execution block to validate taxonomy.json."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    taxonomy_path = os.path.join(current_dir, "taxonomy.json")

    print(f"Starting validation of taxonomy.json at: {taxonomy_path}")
    try:
        taxonomy = load_taxonomy(taxonomy_path)
        validate_structure(taxonomy)
        validate_vocabulary_and_grounding(taxonomy)
        print("Validation Completed: taxonomy.json is VALID!")
    except Exception as e:
        print(f"Validation FAILED: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
