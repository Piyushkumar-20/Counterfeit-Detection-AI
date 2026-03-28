def compute_score(ocr_text, extracted_dosage, candidate):
    score = 0.0

    # 1. Text similarity
    text_score = candidate["score"]  # already 0–1
    score += 0.7 * text_score

    # 2. Dosage match
    db_dosage = candidate.get("dosage")  # you must store this in DB
    if extracted_dosage and db_dosage:
        if extracted_dosage == db_dosage:
            score += 0.3
        else:
            score -= 0.2  # strong penalty

    return score