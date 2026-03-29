import json
import os
import time
import uuid
from typing import Dict


REVIEW_QUEUE_PATH = "database/canonical/review_queue.json"


def _load_queue(path: str = REVIEW_QUEUE_PATH):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data if isinstance(data, list) else []


def _save_queue(entries, path: str = REVIEW_QUEUE_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(entries, file, ensure_ascii=False, indent=2)


def enqueue_for_review(result: Dict, reason: str, queue_path: str = REVIEW_QUEUE_PATH) -> Dict:
    entries = _load_queue(queue_path)
    decision = result.get("decision", {}) if isinstance(result, dict) else {}

    row = {
        "id": str(uuid.uuid4()),
        "created_at": int(time.time()),
        "reason": reason,
        "drug_name": decision.get("drug_name"),
        "final_decision": decision.get("final_decision"),
        "confidence": decision.get("confidence"),
        "probability_authentic": decision.get("probability_authentic"),
        "feature_breakdown": decision.get("feature_breakdown", {}),
        "regulatory_assessment": decision.get("regulatory_assessment", {}),
        "status": "pending",
    }
    entries.append(row)

    # Keep queue bounded to avoid storage growth.
    if len(entries) > 2000:
        entries = sorted(entries, key=lambda item: item.get("created_at", 0), reverse=True)[:2000]

    _save_queue(entries, queue_path)
    return row
