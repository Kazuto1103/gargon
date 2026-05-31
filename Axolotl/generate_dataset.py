"""Dataset Generation Module for Grounded NLP.

This script programmatically generates a balanced synthetic dataset of over 2000
unique Indonesian commands labeled with Intents and BIO-tagged slots.
"""

import json
import os
import random
import re
from typing import Any, Dict, List, Set, Tuple


def tokenize_sentence(sentence: str) -> List[str]:
    """Tokenizes a sentence by splitting on whitespace and removing punctuation.

    Args:
        sentence (str): The raw text sentence.

    Returns:
        List[str]: The list of tokens.
    """
    cleaned = re.sub(r"[.,\/#!$%\^&\*;:{}=\-_`~()?]", " ", sentence)
    return [t for t in cleaned.split() if t]


# Core closed-domain vocabulary definition (aligned with taxonomy.json)
VOCABULARY: Dict[str, List[str]] = {
    "STOP_ACTION": ["berhenti", "setop", "stop", "diam", "tahan", "hentikan", "batal", "batalkan"],
    "STANDARD_ACTION": [
        "maju", "gerak", "jalan", "meluncur", "geser", "bergeser",
        "mundur", "kembali", "kebelakang",
        "putar", "belok", "berputar", "memutar", "belokkan",
        "angkat", "naikkan", "turunkan", "turun", "naik",
        "ambil", "jepit", "pegang", "cengkeram", "tangkap",
        "lepas", "lepaskan", "taruh", "letakkan"
    ],
    "DIRECTION": [
        "kiri", "mengiri", "ke kiri", "sebelah kiri",
        "kanan", "menganan", "ke kanan", "sebelah kanan",
        "depan", "ke depan", "sebelah depan", "lurus",
        "belakang", "ke belakang", "sebelah belakang",
        "atas", "ke atas", "sebelah atas",
        "bawah", "ke bawah", "sebelah bawah",
        "searah jarum jam", "berlawanan jarum jam",
        "balik", "putar balik", "kembali"
    ],
    "QUANTITY": [
        "nol", "kosong", "satu", "dua", "tiga", "empat", "lima", "enam", "tujuh", "delapan", "sembilan",
        "sepuluh", "sebelas", "dua belas", "setengah", "separuh", "seperempat",
        "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "15", "30", "45", "50", "90", "180", "360", "0.5", "1.5", "2.5"
    ],
    "UNIT": [
        "meter", "m", "centimeter", "cm", "senti", "mili", "milimeter", "mm",
        "derajat", "deg", "derajad", "putaran", "langkah",
        "kali", "x",
        "detik", "sekon", "s", "menit", "mnt", "jam", "hari"
    ],
    "MODIFIER": [
        "cepat", "kencang", "ngebut", "cepet", "maksimal", "penuh",
        "lambat", "pelan", "perlahan", "alon-alon", "pelan-pelan",
        "sedang", "normal", "biasa",
        "darurat", "sekarang juga", "secepatnya"
    ],
    "TIME": [
        "pukul 12", "jam 3 sore", "nanti malam", "pukul 10 pagi", "jam 9 pagi", "jam 5 sore", "pukul delapan",
        "pukul 15:00", "jam 10 pagi", "jam 12 siang", "pukul 08:00", "jam 1", "jam 2 siang", "pukul 18:30",
        "nanti jam 3 sore", "pukul delapan malam", "nanti pukul 12", "jam sebelas malam", "jam 12 malam",
        "pukul 6 pagi", "pukul 7 pagi", "pukul 12:00", "pukul 20:00", "jam 8 malam"
    ],
    "DATE": [
        "hari ini", "besok", "lusa", "kemarin", "senin depan", "minggu depan", "31 mei", "tanggal 1",
        "bulan depan", "besok lusa", "senin", "selasa", "rabu", "kamis", "jumat", "sabtu", "minggu"
    ],
    # Unknown/Casual vocabulary
    "CASUAL_SUBJECT": ["saya", "kamu", "dia", "mereka", "kami", "kita", "bapak", "ibu", "teman saya", "anak itu"],
    "CASUAL_VERB": ["ingin", "sedang", "suka", "mau", "perlu", "bisa", "belajar", "melihat", "membeli", "mencari"],
    "CASUAL_ACTIVITY": [
        "membaca buku", "menonton film", "makan nasi goreng", "minum kopi hangat",
        "bermain game", "menulis skrip python", "mendengarkan musik", "pergi ke kantor",
        "tidur siang", "memasak makan malam"
    ],
    "CASUAL_ADVERB": [
        "sekarang", "hari ini", "kemarin", "besok", "di rumah", "di kantor",
        "bersama keluarga", "dengan senang", "sambil santai"
    ]
}


# Templates per Intent
TEMPLATES: Dict[str, List[List[str]]] = {
    "DIRECT_COMMAND": [
        ["{STANDARD_ACTION}", "{DIRECTION}", "{QUANTITY}", "{UNIT}"],
        ["tolong", "{STANDARD_ACTION}", "{DIRECTION}", "{QUANTITY}", "{UNIT}"],
        ["{STANDARD_ACTION}", "{DIRECTION}", "{QUANTITY}", "{UNIT}", "dengan", "{MODIFIER}"],
        ["segera", "{STANDARD_ACTION}", "{DIRECTION}"],
        ["{STANDARD_ACTION}", "{DIRECTION}", "{MODIFIER}"],
        ["{STANDARD_ACTION}", "{QUANTITY}", "{UNIT}", "{DIRECTION}"],
        ["tolong", "{STANDARD_ACTION}", "{QUANTITY}", "{UNIT}", "{DIRECTION}"],
        ["{STANDARD_ACTION}", "{DIRECTION}", "{QUANTITY}", "{UNIT}", "{MODIFIER}"],
        ["segera", "{STANDARD_ACTION}", "{QUANTITY}", "{UNIT}", "{DIRECTION}", "secara", "{MODIFIER}"],
        ["{STANDARD_ACTION}", "{DIRECTION}"]
    ],
    "SCHEDULED_COMMAND": [
        ["{STANDARD_ACTION}", "{DIRECTION}", "{QUANTITY}", "{UNIT}", "pada", "{DATE}", "{TIME}"],
        ["{STANDARD_ACTION}", "{DIRECTION}", "{QUANTITY}", "{UNIT}", "{TIME}"],
        ["pada", "{DATE}", "{TIME}", "tolong", "{STANDARD_ACTION}", "{DIRECTION}", "{QUANTITY}", "{UNIT}"],
        ["{DATE}", "{TIME}", "lakukan", "{STANDARD_ACTION}", "{DIRECTION}", "{QUANTITY}", "{UNIT}"],
        ["tolong", "{STANDARD_ACTION}", "{DIRECTION}", "{QUANTITY}", "{UNIT}", "{TIME}"],
        ["{STANDARD_ACTION}", "{DIRECTION}", "{TIME}"],
        ["{STANDARD_ACTION}", "{DIRECTION}", "pada", "{DATE}"],
        ["segera", "{STANDARD_ACTION}", "{DIRECTION}", "setelah", "{TIME}"],
        ["{STANDARD_ACTION}", "{DIRECTION}", "{QUANTITY}", "{UNIT}", "dengan", "{MODIFIER}", "pada", "{DATE}", "{TIME}"],
        ["{STANDARD_ACTION}", "{QUANTITY}", "{UNIT}", "pada", "{TIME}"]
    ],
    "REPEATED_COMMAND": [
        ["{STANDARD_ACTION}", "{DIRECTION}", "{QUANTITY}", "{UNIT}", "setiap", "{QUANTITY}", "{UNIT}"],
        ["{STANDARD_ACTION}", "{DIRECTION}", "setiap", "{QUANTITY}", "{UNIT}"],
        ["setiap", "{QUANTITY}", "{UNIT}", "tolong", "{STANDARD_ACTION}", "{DIRECTION}"],
        ["tiap", "{QUANTITY}", "{UNIT}", "{STANDARD_ACTION}", "{DIRECTION}", "{QUANTITY}", "{UNIT}"],
        ["{STANDARD_ACTION}", "{DIRECTION}", "{QUANTITY}", "{UNIT}", "tiap", "{UNIT}"],
        ["{STANDARD_ACTION}", "{DIRECTION}", "setiap", "{UNIT}"],
        ["tiap", "{UNIT}", "lakukan", "{STANDARD_ACTION}", "{DIRECTION}"],
        ["{STANDARD_ACTION}", "{DIRECTION}", "dengan", "{MODIFIER}", "setiap", "{QUANTITY}", "{UNIT}"],
        ["setiap", "{QUANTITY}", "{UNIT}", "{STANDARD_ACTION}", "{DIRECTION}", "dengan", "{MODIFIER}"],
        ["{STANDARD_ACTION}", "{DIRECTION}", "{QUANTITY}", "{UNIT}", "setiap", "{QUANTITY}", "{UNIT}", "mulai", "{TIME}"]
    ],
    "STOP_COMMAND": [
        ["{STOP_ACTION}"],
        ["{STOP_ACTION}", "sekarang"],
        ["{STOP_ACTION}", "robot"],
        ["tolong", "{STOP_ACTION}"],
        ["tolong", "{STOP_ACTION}", "semua", "gerakan"],
        ["{STOP_ACTION}", "secepatnya"],
        ["{STOP_ACTION}", "sekarang", "juga"],
        ["segera", "{STOP_ACTION}"],
        ["{STOP_ACTION}", "semua", "jadwal"],
        ["{STOP_ACTION}", "aksi", "ini"],
        ["{STOP_ACTION}", "{STANDARD_ACTION}"],
        ["{STOP_ACTION}", "{STANDARD_ACTION}", "{DIRECTION}"],
        ["{STOP_ACTION}", "{STANDARD_ACTION}", "{DIRECTION}", "{DATE}"],
        ["{STOP_ACTION}", "{STANDARD_ACTION}", "pada", "{DATE}", "{TIME}"]
    ],
    "UNKNOWN": [
        ["{CASUAL_SUBJECT}", "{CASUAL_VERB}", "{CASUAL_ACTIVITY}"],
        ["{CASUAL_SUBJECT}", "{CASUAL_VERB}", "{CASUAL_ACTIVITY}", "{CASUAL_ADVERB}"],
        ["tolong", "bantu", "{CASUAL_SUBJECT}", "untuk", "{CASUAL_ACTIVITY}"],
        ["bagaimana", "cara", "{CASUAL_ACTIVITY}"],
        ["apakah", "{CASUAL_SUBJECT}", "bisa", "{CASUAL_ACTIVITY}"],
        ["cuaca", "{CASUAL_ADVERB}", "sangat", "cerah"],
        ["{CASUAL_SUBJECT}", "sedang", "sibuk", "{CASUAL_ADVERB}"],
        ["halo", "apa", "kabar", "teman", "semua"],
        ["terima", "kasih", "atas", "bantuannya"]
    ]
}


def fill_template(template: List[str]) -> Tuple[List[str], List[str]]:
    """Fills placeholders in a template and returns tokenized sentence and BIO tags.

    Args:
        template (List[str]): A list representing sentence segment templates.

    Returns:
        Tuple[List[str], List[str]]: A tuple of (tokens, BIO slots).
    """
    tokens: List[str] = []
    slots: List[str] = []
    last_slot_type: str = "O"

    for part in template:
        # Check if part is a placeholder
        if part.startswith("{") and part.endswith("}"):
            vocab_key = part[1:-1]
            # Standard actions and Stop actions map to ACTION slot
            if vocab_key in ("STANDARD_ACTION", "STOP_ACTION"):
                slot_type = "ACTION"
            elif vocab_key == "QUANTITY_WORDS":
                slot_type = "QUANTITY"
            else:
                slot_type = vocab_key

            phrase = random.choice(VOCABULARY[vocab_key])
            phrase_tokens = tokenize_sentence(phrase)

            for i, token in enumerate(phrase_tokens):
                # Handle slot mapping for BIO tagging
                # If continuing the same slot type (e.g. contiguous ACTION or multi-word)
                if i == 0 and last_slot_type != slot_type:
                    tag = f"B-{slot_type}"
                else:
                    tag = f"I-{slot_type}"

                tokens.append(token)
                slots.append(tag)

            last_slot_type = slot_type
        else:
            # Plain filler word
            filler_tokens = tokenize_sentence(part)
            for token in filler_tokens:
                tokens.append(token)
                slots.append("O")
            last_slot_type = "O"

    return tokens, slots


def generate_dataset(target_size_per_intent: int = 500) -> List[Dict[str, Any]]:
    """Generates a balanced dataset of unique sentences.

    Args:
        target_size_per_intent (int): Target number of unique sentences per intent.

    Returns:
        List[Dict[str, Any]]: The list of dataset objects.
    """
    dataset: List[Dict[str, Any]] = []
    seen_sentences: Set[str] = set()

    for intent, templates in TEMPLATES.items():
        intent_count = 0
        max_attempts = 15000  # Prevent infinite loop if combinations are limited
        attempts = 0

        while intent_count < target_size_per_intent and attempts < max_attempts:
            attempts += 1
            template = random.choice(templates)
            tokens, slots = fill_template(template)
            sentence_str = " ".join(tokens)

            if sentence_str not in seen_sentences:
                seen_sentences.add(sentence_str)

                # For UNKNOWN intent, all slots must be 'O'
                if intent == "UNKNOWN":
                    slots = ["O"] * len(tokens)

                dataset.append({
                    "sentence": sentence_str,
                    "intent": intent,
                    "tokens": tokens,
                    "slots": slots
                })
                intent_count += 1

        print(f"Generated {intent_count} unique sentences for intent '{intent}' (attempts: {attempts})")

    # Shuffle the dataset to mix intents
    random.shuffle(dataset)
    return dataset


def main() -> None:
    """Main execution entry point to generate and save the dataset."""
    random.seed(42)  # Set seed for reproducibility as per coding standards
    print("Generating synthetic dataset...")
    data = generate_dataset(550)  # Generate slightly more than 500 to guarantee balance

    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(current_dir, "synthetic_dataset.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Dataset successfully saved to: {output_path}")
    print(f"Total dataset size: {len(data)} rows")


if __name__ == "__main__":
    main()
