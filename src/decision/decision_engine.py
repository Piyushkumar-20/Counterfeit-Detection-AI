from src.ocr.extractors import extract_dosage


def verify(candidates, extracted_data, uv_features, database, text):

    if not candidates:
        return "UNKNOWN DRUG"

    dosage = extract_dosage(text)

    best = None
    best_score = -1

    for c in candidates:
        name = c["name"]
        score = c["score"]

        db_entry = database.get(name)

        if not db_entry:
            continue

        db_dosage = db_entry.get("dosage")

        # Dosage check
        if dosage and db_dosage:
            if dosage == db_dosage:
                score += 0.2
            else:
                score -= 0.3

        # UV check
        if db_entry.get("uv_required", False):
            if not uv_features["uv_present"]:
                score -= 0.5

        if score > best_score:
            best_score = score
            best = name

    if best_score > 0.7:
        return f"LIKELY GENUINE ({best})"
    else:
        return f"SUSPICIOUS / FAKE ({best})"

def validate_dosage(dosage):
    valid_dosages = [125, 250, 500, 650, 1000]

    if dosage is None:
        return {"status": "missing", "valid": False}

    if dosage in valid_dosages:
        return {"status": "valid", "valid": True}

    return {
        "status": "suspicious",
        "valid": False,
        "reason": f"Unusual dosage: {dosage}"
    }