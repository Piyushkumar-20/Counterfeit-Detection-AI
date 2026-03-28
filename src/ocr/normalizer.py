import re
from src.ocr.rules import NUMERIC_MAP
from src.ocr.patterns import reconstruct_numeric_patterns, validate_numeric_token


# -----------------------------
# BASIC NUMERIC NORMALIZATION
# -----------------------------
def normalize_numeric(token):
    return ''.join(NUMERIC_MAP.get(c, c) for c in token)


# -----------------------------
# DETECT DOSAGE TOKENS
# -----------------------------
def is_dosage_token(token):
    token_lower = token.lower()

    if "mg" in token_lower or "ml" in token_lower:
        return True

    if any(char.isdigit() for char in token):
        return True

    if re.search(r'[0-9oO]{3,}', token):
        return True

    return False


# -----------------------------
# MAIN NORMALIZATION PIPELINE
# -----------------------------
def normalize_ocr_errors(text):
    tokens = text.split()
    normalized_tokens = []

    for token in tokens:

        if is_dosage_token(token):
            corrected = normalize_numeric(token)
            corrected = reconstruct_numeric_patterns(corrected)
            corrected = validate_numeric_token(corrected)
        else:
            corrected = token

        normalized_tokens.append(corrected)

    return " ".join(normalized_tokens)