from rapidfuzz import process, fuzz

def get_best_drug_candidates(text, known_drugs, k=3):
    results = process.extract(
        text,
        known_drugs,
        scorer=fuzz.partial_ratio,
        limit=k
    )

    return [
        {"name": match, "score": score / 100}
        for match, score, _ in results
    ]