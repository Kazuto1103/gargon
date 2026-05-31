"""Grounding Translator Module (Phase 5).

This module implements the deterministic Grounding Engine that receives predicted
intent and CRF (token, BIO-tag) pairs from Phase 4, resolves semantic anomalies
using strict fallback rules, and produces a canonical JSON command payload
ready to be consumed by a robot scheduler or microcontroller.
"""

import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Canonical Lookup Tables (derived from taxonomy.json grounding_rules)
# ---------------------------------------------------------------------------

# Action token -> universal robot command constant
ACTION_MAP: Dict[str, str] = {
    "maju": "MOVE_FORWARD",
    "gerak": "MOVE_FORWARD",
    "jalan": "MOVE_FORWARD",
    "meluncur": "MOVE_FORWARD",
    "geser": "MOVE_FORWARD",
    "bergeser": "MOVE_FORWARD",
    "mundur": "MOVE_BACKWARD",
    "kembali": "MOVE_BACKWARD",
    "kebelakang": "MOVE_BACKWARD",
    "putar": "ROTATE",
    "belok": "ROTATE",
    "berputar": "ROTATE",
    "memutar": "ROTATE",
    "belokkan": "ROTATE",
    "muter": "ROTATE",
    "puter": "ROTATE",
    "angkat": "MOVE_ARM_UP",
    "naikkan": "MOVE_ARM_UP",
    "naik": "MOVE_ARM_UP",
    "turunkan": "MOVE_ARM_DOWN",
    "turun": "MOVE_ARM_DOWN",
    "ambil": "GRAB",
    "jepit": "GRAB",
    "pegang": "GRAB",
    "cengkeram": "GRAB",
    "tangkap": "GRAB",
    "lepas": "RELEASE",
    "lepaskan": "RELEASE",
    "taruh": "RELEASE",
    "letakkan": "RELEASE",
    "berhenti": "STOP",
    "setop": "STOP",
    "stop": "STOP",
    "diam": "STOP",
    "tahan": "STOP",
    "hentikan": "STOP",
    "batal": "STOP",
    "batalkan": "STOP",
}

# Direction token -> universal direction constant
DIRECTION_MAP: Dict[str, str] = {
    "kiri": "LEFT",
    "mengiri": "LEFT",
    "ke": "UNRESOLVED",   # Context-dependent, resolved via next token
    "kanan": "RIGHT",
    "menganan": "RIGHT",
    "depan": "FRONT",
    "lurus": "FRONT",
    "belakang": "BACK",
    "balik": "BACK",
    "atas": "UP",
    "bawah": "DOWN",
    "searah": "CLOCKWISE",
    "berlawanan": "COUNTER_CLOCKWISE",
    "jarum": "CLOCKWISE",     # I- token continuation
    "jam": "CLOCKWISE",       # I- token continuation (searah jarum jam)
}

# Default direction per action constant when direction slot is missing
DEFAULT_DIRECTION: Dict[str, str] = {
    "MOVE_FORWARD": "FRONT",
    "MOVE_BACKWARD": "BACK",
    "ROTATE": "RIGHT",
    "MOVE_ARM_UP": "UP",
    "MOVE_ARM_DOWN": "DOWN",
    "GRAB": "FRONT",
    "RELEASE": "FRONT",
    "STOP": None,
}

# Unit taxonomies as required by Phase 5b
SPATIAL_UNITS = ["METER", "CENTIMETER", "DERAJAT", "STEP"]
TEMPORAL_UNITS = ["MENIT", "JAM", "DETIK", "KALI"]

# Unit token -> canonical unit constant
UNIT_MAP: Dict[str, str] = {
    "meter": "METER",
    "m": "METER",
    "centimeter": "CENTIMETER",
    "cm": "CENTIMETER",
    "senti": "CENTIMETER",
    "milimeter": "CENTIMETER",  # Maps to spatial CENTIMETER
    "mm": "CENTIMETER",
    "mili": "CENTIMETER",
    "derajat": "DERAJAT",
    "deg": "DERAJAT",
    "derajad": "DERAJAT",
    "putaran": "DERAJAT",       # Maps to spatial DERAJAT
    "langkah": "STEP",
    "kali": "KALI",
    "x": "KALI",
    "detik": "DETIK",
    "sekon": "DETIK",
    "s": "DETIK",
    "menit": "MENIT",
    "mnt": "MENIT",
    "jam": "JAM",
    "hari": "JAM",            # Maps to temporal JAM
}

# Textual quantity words to float values
QUANTITY_WORD_MAP: Dict[str, float] = {
    "nol": 0.0,
    "kosong": 0.0,
    "satu": 1.0,
    "dua": 2.0,
    "tiga": 3.0,
    "empat": 4.0,
    "lima": 5.0,
    "enam": 6.0,
    "tujuh": 7.0,
    "delapan": 8.0,
    "sembilan": 9.0,
    "sepuluh": 10.0,
    "sebelas": 11.0,
    "dua belas": 12.0,
    "setengah": 0.5,
    "separuh": 0.5,
    "seperempat": 0.25,
    "sedikit": 0.1,
}

# Temporal keywords that signal scheduling context
TEMPORAL_KEYWORDS = {
    "besok", "lusa", "hari ini", "senin", "selasa", "rabu", "kamis", "jumat",
    "sabtu", "minggu", "pukul", "jam", "nanti", "malam", "pagi", "sore",
    "siang", "subuh", "mulai", "setelah",
}

# Safe default quantity when parsing fails
QUANTITY_FALLBACK: float = 1.0
# Safe default unit when none is detected
UNIT_FALLBACK: str = "METER"


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def parse_quantity(raw_tokens: List[str]) -> Tuple[float, Optional[str]]:
    """Parses a list of quantity-slot tokens into a float value.

    Handles digits, decimals, and Indonesian textual numbers.
    Returns a safe fallback if the token is semantically invalid.

    Args:
        raw_tokens (List[str]): The raw token strings from QUANTITY slots.

    Returns:
        Tuple[float, Optional[str]]: (parsed_value, fallback_reason_or_None)
    """
    joined = " ".join(raw_tokens).lower().strip()

    # Check textual quantity map first (supports multi-word like "dua belas")
    if joined in QUANTITY_WORD_MAP:
        return QUANTITY_WORD_MAP[joined], None

    # Check each individual token
    for tok in raw_tokens:
        tok_lower = tok.lower()
        if tok_lower in QUANTITY_WORD_MAP:
            return QUANTITY_WORD_MAP[tok_lower], None

    # Try direct numeric parsing
    try:
        return float(joined), None
    except ValueError:
        pass

    # Try the first token as a float
    for tok in raw_tokens:
        try:
            return float(tok), None
        except ValueError:
            continue

    # All parsing failed: return safe fallback and reason string
    reason = f"QUANTITY_PARSE_FAILED: unrecognized value '{joined}', defaulting to {QUANTITY_FALLBACK}"
    return QUANTITY_FALLBACK, reason


def resolve_direction(direction_tokens: List[str]) -> str:
    """Resolves a list of direction tokens into a canonical direction constant.

    Args:
        direction_tokens (List[str]): The token strings from DIRECTION slots.

    Returns:
        str: The canonical direction constant.
    """
    phrase = " ".join(direction_tokens).lower()

    # Multi-word special cases
    if "searah" in phrase and ("jarum" in phrase or "jam" in phrase):
        return "CLOCKWISE"
    if "berlawanan" in phrase:
        return "COUNTER_CLOCKWISE"
    if "ke kiri" in phrase or "kiri" in phrase:
        return "LEFT"
    if "ke kanan" in phrase or "kanan" in phrase:
        return "RIGHT"
    if "ke depan" in phrase or "depan" in phrase or "lurus" in phrase:
        return "FRONT"
    if "ke belakang" in phrase or "belakang" in phrase or "balik" in phrase:
        return "BACK"
    if "ke atas" in phrase or "atas" in phrase:
        return "UP"
    if "ke bawah" in phrase or "bawah" in phrase:
        return "DOWN"

    # Single token fallback via DIRECTION_MAP
    for tok in direction_tokens:
        result = DIRECTION_MAP.get(tok.lower())
        if result and result != "UNRESOLVED":
            return result

    return "UNRESOLVED"


def resolve_action(action_tokens: List[str]) -> str:
    """Resolves a list of action tokens into a canonical action constant.

    Args:
        action_tokens (List[str]): The token strings from ACTION slots.

    Returns:
        str: The canonical action constant.
    """
    for tok in action_tokens:
        result = ACTION_MAP.get(tok.lower())
        if result:
            return result
    return "UNKNOWN_ACTION"


def resolve_unit(unit_tokens: List[str]) -> str:
    """Resolves a list of unit tokens into a canonical unit constant.

    Args:
        unit_tokens (List[str]): The token strings from UNIT slots.

    Returns:
        str: The canonical unit constant.
    """
    for tok in unit_tokens:
        result = UNIT_MAP.get(tok.lower())
        if result:
            return result
    return UNIT_FALLBACK


def extract_slots_by_type(
    token_tag_pairs: List[Tuple[str, str]]
) -> Dict[str, List[str]]:
    """Groups tokens by their BIO slot type.

    Args:
        token_tag_pairs (List[Tuple[str, str]]): List of (token, BIO-tag) pairs.

    Returns:
        Dict[str, List[str]]: Mapping from slot type to list of token strings.
    """
    slots: Dict[str, List[str]] = {
        "ACTION": [], "DIRECTION": [], "QUANTITY": [],
        "UNIT": [], "MODIFIER": [], "TIME": [], "DATE": [],
    }
    for token, tag in token_tag_pairs:
        if tag == "O":
            continue
        # Strip B- or I- prefix to get slot type
        if tag.startswith("B-") or tag.startswith("I-"):
            slot_type = tag[2:]
            if slot_type in slots:
                slots[slot_type].append(token)
    return slots


def build_temporal_context(
    time_tokens: List[str],
    date_tokens: List[str],
    intent: str,
) -> Dict[str, Any]:
    """Builds the temporal scheduling context from TIME and DATE slot tokens.

    Args:
        time_tokens (List[str]): Tokens from TIME slots.
        date_tokens (List[str]): Tokens from DATE slots.
        intent (str): The predicted intent string.

    Returns:
        Dict[str, Any]: Temporal context sub-dictionary.
    """
    is_scheduled = intent in ("SCHEDULED_COMMAND", "REPEATED_COMMAND")
    execute_at: Optional[str] = None

    if time_tokens or date_tokens:
        parts = []
        if date_tokens:
            parts.append(" ".join(date_tokens))
        if time_tokens:
            parts.append(" ".join(time_tokens))
        execute_at = ", ".join(parts) if parts else None

    return {
        "is_scheduled": is_scheduled,
        "execute_at": execute_at,
    }


# ---------------------------------------------------------------------------
# Grounding Engine
# ---------------------------------------------------------------------------

class GroundingEngine:
    """Translates SVM intent + CRF slot predictions into a canonical JSON command.

    This engine applies deterministic, prioritized rules to resolve semantic
    anomalies and produce a well-structured, type-safe output payload.
    """

    def translate(
        self,
        intent: str,
        token_tag_pairs: List[Tuple[str, str]],
    ) -> Dict[str, Any]:
        """Main entry point: ground a predicted intent and slot sequence.

        Args:
            intent (str): The SVM-predicted intent label.
            token_tag_pairs (List[Tuple[str, str]]): List of (token, BIO-tag)
                from the CRF slot filler.

        Returns:
            Dict[str, Any]: Canonical JSON-serializable command payload.
        """
        fallback_triggered = False
        fallback_reason: Optional[str] = None

        # ---------------------------------------------------------------
        # RULE 1: INTENT SUPREMACY — Emergency Brake for STOP_COMMAND
        # ---------------------------------------------------------------
        if intent == "STOP_COMMAND":
            return {
                "status": "SUCCESS",
                "command": {
                    "intent": "STOP_COMMAND",
                    "action": "STOP",
                    "type": "EMERGENCY_BRAKE",
                },
                "parameters": {
                    "spatial": {
                        "direction": None,
                        "quantity": None,
                        "unit": None,
                    },
                    "temporal": {
                        "is_scheduled": False,
                        "execute_at": None,
                        "interval_quantity": None,
                        "interval_unit": None,
                    },
                },
                "pipeline_metadata": {
                    "fallback_triggered": False,
                    "fallback_reason": None,
                },
            }

        # ---------------------------------------------------------------
        # RULE: UNKNOWN intent — reject gracefully
        # ---------------------------------------------------------------
        if intent == "UNKNOWN":
            return {
                "status": "REJECTED",
                "command": {
                    "intent": "UNKNOWN",
                    "action": None,
                    "type": None,
                },
                "parameters": {
                    "spatial": {
                        "direction": None,
                        "quantity": None,
                        "unit": None,
                    },
                    "temporal": {
                        "is_scheduled": False,
                        "execute_at": None,
                        "interval_quantity": None,
                        "interval_unit": None,
                    },
                },
                "pipeline_metadata": {
                    "fallback_triggered": True,
                    "fallback_reason": "REJECTED: Intent classified as UNKNOWN. No valid command payload generated.",
                },
            }

        # ---------------------------------------------------------------
        # Extract grouped slots from CRF output
        # ---------------------------------------------------------------
        slots = extract_slots_by_type(token_tag_pairs)

        # ---------------------------------------------------------------
        # RULE 2: Canonical Action Resolution
        # ---------------------------------------------------------------
        canonical_action: str = "UNKNOWN_ACTION"
        if slots["ACTION"]:
            canonical_action = resolve_action(slots["ACTION"])
        else:
            # Action slot is empty: attempt direct ACTION_MAP lookup on all O-tagged tokens
            for tok, tag in token_tag_pairs:
                if tok.lower() in ACTION_MAP:
                    canonical_action = ACTION_MAP[tok.lower()]
                    fallback_triggered = True
                    fallback_reason = (
                        f"ACTION_FALLBACK: No B-ACTION slot found; resolved '{tok}' "
                        f"via direct ACTION_MAP lookup -> '{canonical_action}'"
                    )
                    break

        # ---------------------------------------------------------------
        # RULE 2: Canonical Direction Resolution + default if missing
        # ---------------------------------------------------------------
        canonical_direction: Optional[str] = None
        if slots["DIRECTION"]:
            canonical_direction = resolve_direction(slots["DIRECTION"])
        else:
            # Apply default direction based on action constant
            canonical_direction = DEFAULT_DIRECTION.get(canonical_action)
            if canonical_direction is not None:
                fallback_triggered = True
                fallback_reason = (
                    f"DIRECTION_FALLBACK: No DIRECTION slot found for action "
                    f"'{canonical_action}'. Assigned default direction '{canonical_direction}'."
                )

        # ---------------------------------------------------------------
        # RULE 3: Deterministic Token-Unit Binding & Quantity Parsing (Phase 5b)
        # ---------------------------------------------------------------
        spatial_quantity: Optional[float] = None
        spatial_unit: Optional[str] = None
        temporal_interval_quantity: Optional[float] = None
        temporal_interval_unit: Optional[str] = None

        i = 0
        n = len(token_tag_pairs)
        while i < n:
            tok, tag = token_tag_pairs[i]
            if tag in ("B-QUANTITY", "I-QUANTITY"):
                # Collect the full contiguous block of QUANTITY tokens
                qty_tokens = [tok]
                j = i + 1
                while j < n and token_tag_pairs[j][1] in ("B-QUANTITY", "I-QUANTITY"):
                    qty_tokens.append(token_tag_pairs[j][0])
                    j += 1
                
                # Parse quantity safely
                parsed_val, qty_reason = parse_quantity(qty_tokens)
                if qty_reason:
                    fallback_triggered = True
                    fallback_reason = (
                        (fallback_reason + " | " if fallback_reason else "") + qty_reason
                    )
                
                # Look ahead to find its corresponding 'B-UNIT' (or 'I-UNIT')
                unit_tokens = []
                k = j
                while k < n:
                    utok, utag = token_tag_pairs[k]
                    if utag in ("B-UNIT", "I-UNIT"):
                        # Collect the full contiguous block of UNIT tokens starting here
                        unit_tokens.append(utok)
                        m = k + 1
                        while m < n and token_tag_pairs[m][1] in ("B-UNIT", "I-UNIT"):
                            unit_tokens.append(token_tag_pairs[m][0])
                            m += 1
                        break
                    k += 1
                
                if unit_tokens:
                    resolved_unit_val = resolve_unit(unit_tokens)
                    # Check taxonomy of the standardized unit string
                    if resolved_unit_val in SPATIAL_UNITS:
                        spatial_quantity = parsed_val
                        spatial_unit = resolved_unit_val
                    elif resolved_unit_val in TEMPORAL_UNITS:
                        temporal_interval_quantity = parsed_val
                        temporal_interval_unit = resolved_unit_val
                    else:
                        # Unrecognized unit taxonomy: default to spatial
                        spatial_quantity = parsed_val
                        spatial_unit = resolved_unit_val
                else:
                    # No unit found looking ahead. Determine based on intent.
                    if intent == "REPEATED_COMMAND":
                        temporal_interval_quantity = parsed_val
                        temporal_interval_unit = "MENIT"
                    else:
                        spatial_quantity = parsed_val
                        spatial_unit = "METER"
                
                i = j
            else:
                i += 1

        # Fallbacks for missing spatial parameters if action is a movement action and intent is valid
        if canonical_action != "STOP" and intent != "UNKNOWN":
            if spatial_quantity is None:
                spatial_quantity = QUANTITY_FALLBACK
                fallback_triggered = True
                fallback_reason = (
                    (fallback_reason + " | " if fallback_reason else "") +
                    f"QUANTITY_FALLBACK: No spatial quantity found, defaulting to {QUANTITY_FALLBACK}"
                )
            if spatial_unit is None:
                spatial_unit = UNIT_FALLBACK
                fallback_triggered = True
                fallback_reason = (
                    (fallback_reason + " | " if fallback_reason else "") +
                    f"UNIT_FALLBACK: No spatial unit found, defaulting to {UNIT_FALLBACK}"
                )

        # ---------------------------------------------------------------
        # RULE: Temporal Scheduling Context
        # ---------------------------------------------------------------
        temporal = build_temporal_context(slots["TIME"], slots["DATE"], intent)

        return {
            "status": "SUCCESS",
            "command": {
                "intent": intent,
                "action": canonical_action,
            },
            "parameters": {
                "spatial": {
                    "direction": canonical_direction,
                    "quantity": spatial_quantity,
                    "unit": spatial_unit,
                },
                "temporal": {
                    "is_scheduled": temporal["is_scheduled"],
                    "execute_at": temporal["execute_at"],
                    "interval_quantity": temporal_interval_quantity,
                    "interval_unit": temporal_interval_unit,
                },
            },
            "pipeline_metadata": {
                "fallback_triggered": fallback_triggered,
                "fallback_reason": fallback_reason,
            },
        }


# ---------------------------------------------------------------------------
# Standalone Verification Harness
# ---------------------------------------------------------------------------

def _print_case(title: str, result: Dict[str, Any]) -> None:
    """Prints a formatted grounding result.

    Args:
        title (str): Description of the test case.
        result (Dict[str, Any]): The grounding payload.
    """
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print()


def main() -> None:
    """Standalone verification harness demonstrating grounding edge cases."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(current_dir, "log")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "grounding_translator.log")

    # Replicate DualWriter pattern inline without importing train_models
    original_stdout = sys.stdout
    log_file = open(log_path, "w", encoding="utf-8")

    class _Tee:
        def write(self, msg: str) -> None:
            original_stdout.write(msg)
            log_file.write(msg)
            log_file.flush()

        def flush(self) -> None:
            original_stdout.flush()
            log_file.flush()

    sys.stdout = _Tee()

    engine = GroundingEngine()

    print("=" * 60)
    print("  GROUNDING ENGINE VERIFICATION HARNESS")
    print("=" * 60)
    print()

    # -------- Case 1: STOP_COMMAND with all 'O' tags (stress test #15 anomaly) --------
    _print_case(
        "Case 1 – STOP with all-O CRF tags (emergency brake bypass)",
        engine.translate(
            intent="STOP_COMMAND",
            token_tag_pairs=[
                ("woi", "O"),
                ("robot", "O"),
                ("berhenti", "O"),
                ("secepatnya", "O"),
                ("dong", "O"),
            ],
        ),
    )

    # -------- Case 2: Valid movement command with textual numbers --------
    _print_case(
        "Case 2 – Valid DIRECT_COMMAND: 'maju dua meter' (textual quantity)",
        engine.translate(
            intent="DIRECT_COMMAND",
            token_tag_pairs=[
                ("maju", "B-ACTION"),
                ("dua", "B-QUANTITY"),
                ("meter", "B-UNIT"),
            ],
        ),
    )

    # -------- Case 3: SCHEDULED_COMMAND with time slot --------
    _print_case(
        "Case 3 – Valid SCHEDULED_COMMAND: 'putar ke kiri 90 derajat besok jam 3 sore'",
        engine.translate(
            intent="SCHEDULED_COMMAND",
            token_tag_pairs=[
                ("putar", "B-ACTION"),
                ("ke", "B-DIRECTION"),
                ("kiri", "I-DIRECTION"),
                ("90", "B-QUANTITY"),
                ("derajat", "B-UNIT"),
                ("besok", "B-DATE"),
                ("jam", "B-TIME"),
                ("3", "I-TIME"),
                ("sore", "I-TIME"),
            ],
        ),
    )

    # -------- Case 4: Anomaly - quantity slot contains non-numeric token 'naik' --------
    _print_case(
        "Case 4 – ANOMALY: QUANTITY slot contains non-numeric 'naik' (safe fallback)",
        engine.translate(
            intent="DIRECT_COMMAND",
            token_tag_pairs=[
                ("angkat", "B-ACTION"),
                ("naik", "B-QUANTITY"),   # CRF hallucination: 'naik' tagged as QUANTITY
                ("meter", "B-UNIT"),
            ],
        ),
    )

    # -------- Case 5: Missing direction slot, fallback default applies --------
    _print_case(
        "Case 5 – FALLBACK: DIRECT_COMMAND 'maju 5 meter' with no DIRECTION slot",
        engine.translate(
            intent="DIRECT_COMMAND",
            token_tag_pairs=[
                ("maju", "B-ACTION"),
                ("5", "B-QUANTITY"),
                ("meter", "B-UNIT"),
            ],
        ),
    )

    # -------- Case 6: REPEATED_COMMAND --------
    _print_case(
        "Case 6 – REPEATED_COMMAND: 'belok ke kanan setiap 5 menit'",
        engine.translate(
            intent="REPEATED_COMMAND",
            token_tag_pairs=[
                ("belok", "B-ACTION"),
                ("ke", "B-DIRECTION"),
                ("kanan", "I-DIRECTION"),
                ("5", "B-QUANTITY"),
                ("menit", "B-UNIT"),
            ],
        ),
    )

    # -------- Case 7: UNKNOWN intent, graceful rejection --------
    _print_case(
        "Case 7 – REJECTED: UNKNOWN intent 'hari ini saya ingin makan'",
        engine.translate(
            intent="UNKNOWN",
            token_tag_pairs=[
                ("hari", "O"),
                ("ini", "O"),
                ("saya", "O"),
                ("ingin", "O"),
                ("makan", "O"),
            ],
        ),
    )

    # -------- Case 8: Multi-slot REPEATED_COMMAND (Phase 5b verification) --------
    _print_case(
        "Case 8 – Multi-slot REPEATED_COMMAND: 'tiap 5 menit belok ke kanan 90 derajat'",
        engine.translate(
            intent="REPEATED_COMMAND",
            token_tag_pairs=[
                ("tiap", "O"),
                ("5", "B-QUANTITY"),
                ("menit", "B-UNIT"),
                ("belok", "B-ACTION"),
                ("ke", "B-DIRECTION"),
                ("kanan", "I-DIRECTION"),
                ("90", "B-QUANTITY"),
                ("derajat", "B-UNIT"),
            ],
        ),
    )

    print("Grounding Verification complete! Log saved to:", log_path)
    sys.stdout = original_stdout
    log_file.close()


if __name__ == "__main__":
    main()
