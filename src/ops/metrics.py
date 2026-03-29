import json
import os
import time
from typing import Dict


CANONICAL_DIR = "database/canonical"


def _load_json(path: str, default):
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _count_json_rows(path: str) -> int:
    data = _load_json(path, [])
    return len(data) if isinstance(data, list) else 0


def collect_operational_metrics() -> Dict:
    products_path = os.path.join(CANONICAL_DIR, "products.json")
    events_path = os.path.join(CANONICAL_DIR, "regulatory_events.json")
    snapshots_path = os.path.join(CANONICAL_DIR, "source_snapshots.json")
    quality_path = os.path.join(CANONICAL_DIR, "quality_report.json")
    review_queue_path = os.path.join(CANONICAL_DIR, "review_queue.json")

    quality = _load_json(quality_path, {})
    snapshots = _load_json(snapshots_path, [])

    source_errors = 0
    if isinstance(snapshots, list):
        source_errors = sum(1 for row in snapshots if isinstance(row, dict) and row.get("status") == "error")

    return {
        "generated_at": int(time.time()),
        "products_count": _count_json_rows(products_path),
        "regulatory_events_count": _count_json_rows(events_path),
        "review_queue_count": _count_json_rows(review_queue_path),
        "schema_error_count": int(quality.get("schema_error_count", 0)) if isinstance(quality, dict) else 0,
        "source_error_count": source_errors,
        "quality_report": quality,
    }
