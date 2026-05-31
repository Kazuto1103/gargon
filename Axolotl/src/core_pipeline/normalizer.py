"""Text Normalization Module for Indonesian NLU Pipeline.

This module provides canonical text normalization for Indonesian robotic command text,
covering abbreviations, typos, merged words (like 'kekiri', '10cm'), and common
chatting slang. It must be applied at both training AND inference time to guarantee
consistent tokenization boundaries.
"""

import re
from typing import List


# Normalization dictionary: non-standard token -> standard token from taxonomy
NORM_DICT = {
    # ---- Movement / Actions ----
    "mju": "maju",
    "puter": "putar",
    "belokkin": "belokkan",
    "ngiri": "ke kiri",
    "kekiri": "ke kiri",
    "kekanan": "ke kanan",
    "kedepan": "ke depan",
    "kebelakang": "ke belakang",
    "keatas": "ke atas",
    "kebawah": "ke bawah",
    "muter": "putar",
    "gerakkan": "gerak",
    "dikit": "sedikit",
    # ---- Direction abbreviations ----
    "dpan": "depan",
    "blkang": "belakang",
    # ---- Stop / Cancel ----
    "batalin": "batalkan",
    "setop": "berhenti",
    "diem": "berhenti",
    "heii": "",
    "dong": "",
    "mang": "",
    "deh": "",
    "lah": "",
    "woi": "",
    "dlu": "dulu",
    "dluu": "dulu",
    # ---- Quantity ----
    "stgah": "separuh",
    "setengahnya": "setengah",
    # ---- Units ----
    "mtr": "meter",
    "mtrs": "meter",
    "senti": "centimeter",
    "cm": "centimeter",
    "deg": "derajat",
    "mnt": "menit",
    # ---- Temporal / Time ----
    "skrg": "sekarang",
    "skrang": "sekarang",
    "ntar": "nanti",
    "tar": "nanti",
    "jm": "jam",
    "jm.": "jam",
    "sor": "sore",
    "sre": "sore",
    "bsk": "besok",
    "bsok": "besok",
    "mlm": "malam",
    "mlam": "malam",
    "pg": "pagi",
    "siang2": "siang",
    "malem": "malam",
    "pagii": "pagi",
}


def normalize_text(text: str) -> str:
    """Normalizes raw Indonesian command text to canonical vocabulary tokens.

    This function performs three passes:
    1. Insert a space between adjacent digits and alphabetical units (e.g., "10cm" -> "10 cm").
    2. Token-level substitution using NORM_DICT for each whitespace-delimited token.
    3. Strip and re-join, removing empty strings from dictionary mappings to empty string.

    Args:
        text (str): The raw input sentence.

    Returns:
        str: The normalized, space-delimited sentence string.
    """
    text = text.lower().strip()

    # Insert space between number-letter boundaries (e.g., "10cm" -> "10 cm", "90deg" -> "90 deg")
    # This targets patterns like <digits><letters> and <letters><digits>
    text = re.sub(r"(\d+)([a-zA-Z]+)", r"\1 \2", text)
    text = re.sub(r"([a-zA-Z]+)(\d+)", r"\1 \2", text)

    # Tokenize by whitespace to apply NORM_DICT token by token
    tokens = text.split()
    normalized_tokens: List[str] = []
    for tok in tokens:
        replacement = NORM_DICT.get(tok, None)
        if replacement is None:
            # Token not in dict: keep as-is
            normalized_tokens.append(tok)
        elif replacement == "":
            # Token maps to empty string: drop it (noise word)
            continue
        else:
            # Token maps to a (possibly multi-word) canonical form
            normalized_tokens.extend(replacement.split())

    return " ".join(normalized_tokens)


def tokenize_sentence(sentence: str) -> List[str]:
    """Tokenizes a sentence by splitting on whitespace and stripping punctuation.

    This is the single canonical tokenizer to be used by all modules
    (preprocess_features, stress_test_models, verify_models).

    Args:
        sentence (str): The raw or already-normalized text sentence.

    Returns:
        List[str]: The list of clean tokens.
    """
    cleaned = re.sub(r"[.,\/#!$%\^&\*;:{}=\-_`~()?]", " ", sentence)
    return [t for t in cleaned.split() if t]
