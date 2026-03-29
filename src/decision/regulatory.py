from typing import Dict, List


def classify_regulatory_risk(
    feature_breakdown: Dict[str, float],
    dosage_validation: Dict,
    qr_result: Dict,
    uv_result: Dict,
    has_candidate: bool,
) -> Dict:
    """
    Map model evidence to regulatory-style categories inspired by field investigation practice.
    """
    drug_match = float(feature_breakdown.get("drug_match_score", 0.0))
    qr_score = float(feature_breakdown.get("qr_validity_score", 0.0))
    uv_score = float(feature_breakdown.get("uv_similarity_score", 0.0))

    rationale: List[str] = []

    if not has_candidate or drug_match < 0.45:
        category = "spurious"
        risk_level = "high"
        legal_sections = ["17B", "27(c)"]
        rationale.append("Drug identity is weak or unverifiable")
    elif dosage_validation.get("status") == "suspicious":
        category = "misbranded"
        risk_level = "medium"
        legal_sections = ["17", "27(d)"]
        rationale.append("Declared dosage pattern appears non-compliant")
    elif qr_score < 0.35 and uv_score < 0.35:
        category = "adulterated_or_tampered"
        risk_level = "high"
        legal_sections = ["17A", "27(a)"]
        rationale.append("Security features are weak or inconsistent")
    elif qr_score < 0.5 or uv_score < 0.5:
        category = "not_of_standard_quality"
        risk_level = "medium"
        legal_sections = ["8", "16", "27(d)"]
        rationale.append("Quality/security checks are below acceptance threshold")
    else:
        category = "compliant"
        risk_level = "low"
        legal_sections = []
        rationale.append("Primary regulatory checks are consistent")

    if qr_result.get("found") and not qr_result.get("decoded"):
        rationale.append("QR was detected but decoding failed")
    if not uv_result.get("available", False):
        rationale.append("UV reference evidence is unavailable")

    return {
        "category": category,
        "risk_level": risk_level,
        "legal_sections": legal_sections,
        "rationale": rationale,
    }
