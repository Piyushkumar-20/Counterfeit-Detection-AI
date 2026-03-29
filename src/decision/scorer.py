from typing import Dict


def dosage_score(extracted_dosage: int, expected_dosage: int) -> float:
    if extracted_dosage is None or expected_dosage is None:
        return 0.35
    return 1.0 if int(extracted_dosage) == int(expected_dosage) else 0.0


def qr_score(qr_payload: Dict) -> float:
    if not qr_payload:
        return 0.0

    if not qr_payload.get("found", False):
        return 0.0

    decode_bonus = 0.4 if qr_payload.get("decoded", False) else 0.0
    format_bonus = 0.6 * float(qr_payload.get("format_score", 0.0))
    return min(max(decode_bonus + format_bonus, 0.0), 1.0)


def uv_score(uv_payload: Dict, uv_required: bool) -> float:
    similarity = float(uv_payload.get("similarity", 0.0)) if uv_payload else 0.0
    if not uv_required:
        return 0.7 + (0.3 * similarity)
    return similarity


def compose_feature_vector(
    ocr_confidence: float,
    drug_match_score: float,
    dosage_match_score: float,
    qr_validity_score: float,
    uv_similarity_score: float,
    image_match_score: float = 0.5,
) -> Dict[str, float]:
    return {
        "ocr_confidence": min(max(float(ocr_confidence), 0.0), 1.0),
        "drug_match_score": min(max(float(drug_match_score), 0.0), 1.0),
        "dosage_match_score": min(max(float(dosage_match_score), 0.0), 1.0),
        "qr_validity_score": min(max(float(qr_validity_score), 0.0), 1.0),
        "uv_similarity_score": min(max(float(uv_similarity_score), 0.0), 1.0),
        "image_match_score": min(max(float(image_match_score), 0.0), 1.0),
    }