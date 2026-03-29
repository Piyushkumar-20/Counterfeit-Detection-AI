from typing import Dict, List

from src.decision.scorer import compose_feature_vector, dosage_score, qr_score, uv_score
from src.decision.regulatory import classify_regulatory_risk
from src.models.classifier import HybridAuthenticityClassifier
from src.ocr.extractors import extract_dosage


def validate_dosage(dosage: int) -> Dict:
    valid_dosages = [125, 250, 500, 650, 1000]

    if dosage is None:
        return {"status": "missing", "valid": False}

    if dosage in valid_dosages:
        return {"status": "valid", "valid": True}

    return {
        "status": "suspicious",
        "valid": False,
        "reason": f"Unusual dosage: {dosage}",
    }


def _build_reasoning(features: Dict[str, float], probability: float, candidate_name: str) -> List[str]:
    notes = []
    if features["ocr_confidence"] < 0.5:
        notes.append("OCR confidence is low")
    if features["drug_match_score"] < 0.6:
        notes.append("Drug-name match is weak")
    if features["dosage_match_score"] < 0.5:
        notes.append("Dosage does not match database expectations")
    if features["qr_validity_score"] < 0.5:
        notes.append("QR data is missing or does not follow expected structure")
    if features["uv_similarity_score"] < 0.5:
        notes.append("UV security pattern similarity is low")
    if features.get("image_match_score", 0.5) < 0.5:
        notes.append("Packaging image similarity to legal dataset is low")

    if not notes:
        notes.append("All major checks are consistent with authentic packaging")

    notes.append(f"Top candidate: {candidate_name}")
    notes.append(f"Model probability: {probability:.3f}")
    return notes


def verify(
    candidates: List[Dict],
    database: Dict,
    text: str,
    ocr_confidence: float,
    qr_result: Dict,
    uv_result: Dict,
    image_match_score: float = 0.5,
    regulatory_sources_result: Dict = None,
    classifier: HybridAuthenticityClassifier = None,
) -> Dict:
    if classifier is None:
        classifier = HybridAuthenticityClassifier()

    if not candidates:
        fallback_features = {
            "ocr_confidence": float(ocr_confidence),
            "drug_match_score": 0.0,
            "dosage_match_score": 0.0,
            "qr_validity_score": float(qr_result.get("format_score", 0.0)),
            "uv_similarity_score": float(uv_result.get("similarity", 0.0)),
        }
        dosage_validation = validate_dosage(extract_dosage(text))
        regulatory = classify_regulatory_risk(
            feature_breakdown=fallback_features,
            dosage_validation=dosage_validation,
            qr_result=qr_result,
            uv_result=uv_result,
            has_candidate=False,
        )
        return {
            "drug_name": None,
            "dosage": extract_dosage(text),
            "final_decision": "counterfeit",
            "confidence": 0.15,
            "probability_authentic": 0.15,
            "reasoning": ["No plausible drug candidate matched OCR output"],
            "feature_breakdown": fallback_features,
            "dosage_validation": dosage_validation,
            "regulatory_assessment": regulatory,
        }

    top = candidates[0]
    drug_name = top["name"]
    db_entry = database.get(drug_name, {})

    extracted_dosage = extract_dosage(text)
    expected_dosage = db_entry.get("dosage")
    uv_required = bool(db_entry.get("uv_required", False))

    features = compose_feature_vector(
        ocr_confidence=ocr_confidence,
        drug_match_score=float(top.get("score", 0.0)),
        dosage_match_score=dosage_score(extracted_dosage, expected_dosage),
        qr_validity_score=qr_score(qr_result),
        uv_similarity_score=uv_score(uv_result, uv_required=uv_required),
        image_match_score=image_match_score,
    )

    probability = classifier.predict_proba(features)
    weighted = classifier.weighted_score(features)
    confidence = (probability * 0.75) + (weighted * 0.25)

    final_decision = "authentic" if probability >= 0.6 else "counterfeit"
    reasoning = _build_reasoning(features, probability, drug_name)
    dosage_validation = validate_dosage(extracted_dosage)
    regulatory = classify_regulatory_risk(
        feature_breakdown=features,
        dosage_validation=dosage_validation,
        qr_result=qr_result,
        uv_result=uv_result,
        has_candidate=True,
    )

    return {
        "drug_name": drug_name,
        "dosage": extracted_dosage,
        "final_decision": final_decision,
        "confidence": round(confidence, 4),
        "probability_authentic": round(probability, 4),
        "reasoning": reasoning,
        "feature_breakdown": features,
        "dosage_validation": dosage_validation,
        "regulatory_assessment": regulatory,
        "regulatory_sources": regulatory_sources_result or {"checked": False, "valid": None, "sources": []},
    }