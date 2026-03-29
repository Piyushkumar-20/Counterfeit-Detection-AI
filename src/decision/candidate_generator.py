from typing import Dict, Iterable, List
from rapidfuzz import fuzz, process


def _build_lookup(database: Dict[str, Dict]) -> Dict[str, str]:
    """Map aliases/synonyms to their canonical drug key."""
    lookup: Dict[str, str] = {}
    for canonical_name, payload in database.items():
        key = canonical_name.strip().lower()
        lookup[key] = canonical_name

        aliases = payload.get("aliases", []) if isinstance(payload, dict) else []
        for alias in aliases:
            alias_key = str(alias).strip().lower()
            if alias_key:
                lookup[alias_key] = canonical_name
    return lookup


def get_best_drug_candidates(text: str, database: Dict[str, Dict], k: int = 5) -> List[Dict]:
    if not text:
        return []

    lookup = _build_lookup(database)
    universe = list(lookup.keys())

    if not universe:
        return []

    fuzzy = process.extract(text.lower(), universe, scorer=fuzz.token_set_ratio, limit=max(k * 2, k))

    ranked = []
    seen = set()
    for alias_match, score, _ in fuzzy:
        canonical = lookup[alias_match]
        if canonical in seen:
            continue
        seen.add(canonical)

        token_score = fuzz.partial_ratio(canonical.lower(), text.lower()) / 100.0
        final_score = ((score / 100.0) * 0.7) + (token_score * 0.3)

        ranked.append(
            {
                "name": canonical,
                "score": min(max(final_score, 0.0), 1.0),
                "matched_alias": alias_match,
            }
        )

    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked[:k]