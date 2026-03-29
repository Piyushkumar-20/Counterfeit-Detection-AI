import json
import os
import urllib.parse
import urllib.request
from typing import Dict, Optional


def _request_json(url: str, timeout: int = 5) -> Dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        payload = response.read().decode("utf-8", errors="ignore")
        return json.loads(payload)


def _verify_from_local_cache(drug_name: Optional[str], cache_path: str = "database/regulatory_cache.json") -> Optional[Dict]:
    if not drug_name:
        return None
    if not os.path.exists(cache_path):
        return None

    try:
        with open(cache_path, "r", encoding="utf-8") as file:
            cache = json.load(file)
    except Exception:
        return None

    key = "".join(ch.lower() if ch.isalnum() else "_" for ch in drug_name).strip("_")
    row = cache.get(key)
    if not isinstance(row, dict):
        return None

    if not row.get("checked", False):
        return {
            "checked": False,
            "valid": None,
            "sources": [{"url": row.get("source", "local_cache"), "reachable": False, "valid": None, "payload": row}],
            "reason": row.get("error", "Local regulatory cache source was not reachable"),
        }

    alerts = row.get("active_alerts", [])
    if alerts:
        return {
            "checked": True,
            "valid": False,
            "sources": [{"url": row.get("source", "local_cache"), "reachable": True, "valid": False, "payload": row}],
            "reason": "Local regulatory cache indicates active alert",
        }

    return {
        "checked": True,
        "valid": True,
        "sources": [{"url": row.get("source", "local_cache"), "reachable": True, "valid": True, "payload": row}],
        "reason": "Verified against local regulatory cache",
    }


def verify_with_regulatory_sources(drug_name: Optional[str], dosage: Optional[int], qr_data: Optional[str]) -> Dict:
    """
    Best-effort verification against configured regulatory endpoints.

    Set REGULATORY_ENDPOINTS as comma-separated URLs in environment.
    Each endpoint should accept query params: drug, dosage, qr
    and ideally return JSON with a boolean-like `valid` field.
    """
    endpoints = [value.strip() for value in os.getenv("REGULATORY_ENDPOINTS", "").split(",") if value.strip()]

    if not endpoints:
        local_result = _verify_from_local_cache(drug_name)
        if local_result is not None:
            return local_result
        return {
            "checked": False,
            "valid": None,
            "sources": [],
            "reason": "No regulatory endpoints configured",
        }

    query = urllib.parse.urlencode(
        {
            "drug": drug_name or "",
            "dosage": dosage or "",
            "qr": qr_data or "",
        }
    )

    results = []
    for endpoint in endpoints:
        url = endpoint + ("&" if "?" in endpoint else "?") + query
        try:
            payload = _request_json(url)
            source_valid = bool(payload.get("valid")) if isinstance(payload, dict) else False
            results.append({"url": endpoint, "reachable": True, "valid": source_valid, "payload": payload})
        except Exception as exc:
            results.append({"url": endpoint, "reachable": False, "valid": False, "error": str(exc)})

    reachable = [row for row in results if row.get("reachable")]
    if not reachable:
        return {
            "checked": True,
            "valid": None,
            "sources": results,
            "reason": "Configured sources unreachable",
        }

    valid = any(row.get("valid") for row in reachable)
    return {
        "checked": True,
        "valid": bool(valid),
        "sources": results,
        "reason": "Verified against configured regulatory sources",
    }
