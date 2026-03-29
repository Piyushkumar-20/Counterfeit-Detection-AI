import json
import os
import urllib.parse
import urllib.request
from typing import Dict, Optional


def _request_json(url: str, timeout: int = 5) -> Dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        payload = response.read().decode("utf-8", errors="ignore")
        return json.loads(payload)


def verify_with_regulatory_sources(drug_name: Optional[str], dosage: Optional[int], qr_data: Optional[str]) -> Dict:
    """
    Best-effort verification against configured regulatory endpoints.

    Set REGULATORY_ENDPOINTS as comma-separated URLs in environment.
    Each endpoint should accept query params: drug, dosage, qr
    and ideally return JSON with a boolean-like `valid` field.
    """
    endpoints = [value.strip() for value in os.getenv("REGULATORY_ENDPOINTS", "").split(",") if value.strip()]

    if not endpoints:
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
