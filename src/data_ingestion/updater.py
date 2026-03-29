import argparse
import hashlib
import json
import os
import re
import time
import urllib.parse
import urllib.request
from typing import Dict, List, Optional, Tuple


OPENFDA_LABEL_ENDPOINT = "https://api.fda.gov/drug/label.json"
OPENFDA_ENFORCEMENT_ENDPOINT = "https://api.fda.gov/drug/enforcement.json"


SOURCE_DEFAULTS = {
    "openfda_label": {
        "enabled": True,
        "type": "api",
        "url": OPENFDA_LABEL_ENDPOINT,
        "confidence": 0.92,
        "retry_count": 2,
        "retry_backoff_sec": 1,
    },
    "openfda_enforcement": {
        "enabled": True,
        "type": "api",
        "url": OPENFDA_ENFORCEMENT_ENDPOINT,
        "confidence": 0.92,
        "retry_count": 2,
        "retry_backoff_sec": 1,
    },
    "india_cdsco_bulletin": {
        "enabled": True,
        "type": "file",
        "url": "database/feeds/india_regulatory_feed.json",
        "confidence": 0.9,
        "retry_count": 0,
        "retry_backoff_sec": 0,
    },
    "manufacturer_feed": {
        "enabled": True,
        "type": "file",
        "url": "database/feeds/manufacturer_feed.json",
        "confidence": 0.95,
        "retry_count": 0,
        "retry_backoff_sec": 0,
    },
    "distributor_feed": {
        "enabled": True,
        "type": "file",
        "url": "database/feeds/distributor_feed.json",
        "confidence": 0.93,
        "retry_count": 0,
        "retry_backoff_sec": 0,
    },
    "manual_curated": {
        "enabled": True,
        "type": "manual",
        "url": "database/drug_db.json",
        "confidence": 0.98,
        "retry_count": 0,
        "retry_backoff_sec": 0,
    },
}


def _load_json(path: str, default):
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _save_json(path: str, data) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def _ensure_source_registry(path: str = "database/source_registry.json") -> Dict[str, Dict]:
    registry = _load_json(path, None)
    if isinstance(registry, dict):
        merged = dict(SOURCE_DEFAULTS)
        for key, value in registry.items():
            if isinstance(value, dict):
                merged[key] = {**merged.get(key, {}), **value}
        _save_json(path, merged)
        return merged

    _save_json(path, SOURCE_DEFAULTS)
    return dict(SOURCE_DEFAULTS)


def _request_json(url: str, timeout: int = 15) -> Dict:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Pharmacy-AI-Updater/1.0",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        payload = response.read().decode("utf-8", errors="ignore")
        return json.loads(payload)


def _request_json_with_retry(url: str, timeout: int, retries: int, backoff_sec: int) -> Dict:
    last_error = None
    attempts = max(retries, 0) + 1
    for attempt in range(attempts):
        try:
            return _request_json(url=url, timeout=timeout)
        except Exception as exc:
            last_error = exc
            if attempt < attempts - 1 and backoff_sec > 0:
                time.sleep(backoff_sec * (attempt + 1))
    raise last_error


def _safe_first(value: Optional[List[str]], default: str = "") -> str:
    if not value:
        return default
    if isinstance(value, list):
        return str(value[0]).strip()
    return str(value).strip()


def _normalize_key(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug or "unknown"


def _compact_unique(values: List[str], limit: int = 25) -> List[str]:
    seen = set()
    out: List[str] = []
    for value in values:
        clean = str(value).strip()
        if not clean:
            continue
        key = clean.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(clean)
        if len(out) >= limit:
            break
    return out


def _extract_mg(text: str) -> Optional[int]:
    if not text:
        return None
    match = re.search(r"(\d{2,4})\s*(mg|milligram)", text.lower())
    if match:
        return int(match.group(1))
    return None


def _build_qr_pattern(brand: str) -> str:
    token = re.sub(r"[^A-Z0-9]", "", brand.upper())
    token = token or "DRUG"
    return rf"({token}|GTIN|BATCH|LOT|EXP)"


def _search_openfda_label(term: str, limit: int = 8) -> List[Dict]:
    query = f'(openfda.brand_name:"{term}"+openfda.generic_name:"{term}")'
    url = f"{OPENFDA_LABEL_ENDPOINT}?search={urllib.parse.quote(query)}&limit={limit}"
    payload = _request_json(url)
    return payload.get("results", []) if isinstance(payload, dict) else []


def _search_openfda_label_with_config(term: str, source_cfg: Dict, limit: int = 8) -> List[Dict]:
    query = f'(openfda.brand_name:"{term}"+openfda.generic_name:"{term}")'
    url = f"{source_cfg.get('url', OPENFDA_LABEL_ENDPOINT)}?search={urllib.parse.quote(query)}&limit={limit}"
    payload = _request_json_with_retry(
        url=url,
        timeout=int(source_cfg.get("timeout_sec", 15)),
        retries=int(source_cfg.get("retry_count", 2)),
        backoff_sec=int(source_cfg.get("retry_backoff_sec", 1)),
    )
    return payload.get("results", []) if isinstance(payload, dict) else []


def _search_openfda_enforcement(term: str, limit: int = 6) -> List[Dict]:
    # Query in product description to capture recall/warning records.
    query = f'product_description:"{term}"'
    url = f"{OPENFDA_ENFORCEMENT_ENDPOINT}?search={urllib.parse.quote(query)}&limit={limit}"
    payload = _request_json(url)
    return payload.get("results", []) if isinstance(payload, dict) else []


def _search_openfda_enforcement_with_config(term: str, source_cfg: Dict, limit: int = 6) -> List[Dict]:
    query = f'product_description:"{term}"'
    url = f"{source_cfg.get('url', OPENFDA_ENFORCEMENT_ENDPOINT)}?search={urllib.parse.quote(query)}&limit={limit}"
    payload = _request_json_with_retry(
        url=url,
        timeout=int(source_cfg.get("timeout_sec", 15)),
        retries=int(source_cfg.get("retry_count", 2)),
        backoff_sec=int(source_cfg.get("retry_backoff_sec", 1)),
    )
    return payload.get("results", []) if isinstance(payload, dict) else []


def _schema_errors(entry_key: str, entry: Dict) -> List[str]:
    errors = []
    required = ["brand", "manufacturer", "aliases", "expected_text_patterns", "qr_format"]
    for field in required:
        if field not in entry:
            errors.append(f"{entry_key}: missing {field}")
            continue
        if isinstance(entry[field], str) and not entry[field].strip():
            errors.append(f"{entry_key}: empty {field}")
        if isinstance(entry[field], list) and len(entry[field]) == 0:
            errors.append(f"{entry_key}: empty {field}")

    if entry.get("dosage") is not None and not isinstance(entry.get("dosage"), int):
        errors.append(f"{entry_key}: dosage must be int or null")

    return errors


def _normalize_label_record(raw: Dict) -> Optional[Tuple[str, Dict, str]]:
    openfda = raw.get("openfda", {}) if isinstance(raw, dict) else {}
    brand = _safe_first(openfda.get("brand_name"))
    generic = _safe_first(openfda.get("generic_name"))
    manufacturer = _safe_first(openfda.get("manufacturer_name"), default="Unknown")

    if not brand and not generic:
        return None

    display_name = brand or generic
    key = _normalize_key(display_name)

    dosage_text = " ".join(raw.get("dosage_and_administration", [])[:1]) if isinstance(raw.get("dosage_and_administration"), list) else ""
    ingredient_text = " ".join(raw.get("active_ingredient", [])[:1]) if isinstance(raw.get("active_ingredient"), list) else ""
    dosage = _extract_mg(dosage_text) or _extract_mg(ingredient_text)

    expected_patterns = _compact_unique(
        [
            display_name,
            generic,
            ingredient_text,
            f"{dosage} mg" if dosage else "",
        ],
        limit=10,
    )

    aliases = _compact_unique(
        [display_name, generic, _safe_first(openfda.get("substance_name"))],
        limit=10,
    )

    normalized = {
        "product_id": _safe_first(openfda.get("product_ndc")) or key,
        "brand": display_name,
        "generic_name": generic or display_name,
        "dosage": dosage,
        "strength_unit": "mg" if dosage else None,
        "dosage_form": _safe_first(openfda.get("dosage_form")) or "unknown",
        "pack_presentation": None,
        "manufacturer": manufacturer,
        "manufacturer_license": None,
        "country": "US",
        "regulator": "FDA",
        "expected_text_patterns": expected_patterns,
        "aliases": aliases,
        "known_ocr_distortions": [],
        "qr_format": _build_qr_pattern(display_name),
        "batch_format": "[A-Z0-9-]{4,20}",
        "expiry_format": "(MM/YYYY|YYYY-MM)",
        "manufacturing_date_format": "(MM/YYYY|YYYY-MM)",
        "uv_required": True,
        "uv_signature": [],
        "reference_images": [],
        "hologram_metadata": {},
        "layout_anchors": {},
        "packaging_versions": [],
        "product_status": "active",
        "source_url": OPENFDA_LABEL_ENDPOINT,
        "source_confidence": 0.92,
        "data_sources": ["openfda_label"],
        "last_refreshed": int(time.time()),
        "last_verified": int(time.time()),
    }

    # Stable hash for dedupe; sort keys for deterministic content hash.
    fingerprint = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return key, normalized, fingerprint


def _merge_entry(existing: Dict, incoming: Dict) -> Dict:
    merged = dict(existing)

    merged["product_id"] = existing.get("product_id") or incoming.get("product_id")
    merged["brand"] = incoming.get("brand") or existing.get("brand")
    merged["generic_name"] = existing.get("generic_name") or incoming.get("generic_name")
    merged["manufacturer"] = incoming.get("manufacturer") or existing.get("manufacturer", "Unknown")
    merged["dosage"] = existing.get("dosage") or incoming.get("dosage")
    merged["strength_unit"] = existing.get("strength_unit") or incoming.get("strength_unit")
    merged["dosage_form"] = existing.get("dosage_form") or incoming.get("dosage_form", "unknown")
    merged["pack_presentation"] = existing.get("pack_presentation") or incoming.get("pack_presentation")
    merged["manufacturer_license"] = existing.get("manufacturer_license") or incoming.get("manufacturer_license")
    merged["country"] = existing.get("country") or incoming.get("country")
    merged["regulator"] = existing.get("regulator") or incoming.get("regulator")

    merged["expected_text_patterns"] = _compact_unique(
        list(existing.get("expected_text_patterns", [])) + list(incoming.get("expected_text_patterns", [])),
        limit=20,
    )
    merged["aliases"] = _compact_unique(
        list(existing.get("aliases", [])) + list(incoming.get("aliases", [])),
        limit=20,
    )

    merged.setdefault("known_ocr_distortions", existing.get("known_ocr_distortions", []))
    merged.setdefault("uv_signature", existing.get("uv_signature", []))
    merged.setdefault("reference_images", existing.get("reference_images", []))
    merged.setdefault("hologram_metadata", existing.get("hologram_metadata", {}))
    merged.setdefault("layout_anchors", existing.get("layout_anchors", {}))
    merged.setdefault("packaging_versions", existing.get("packaging_versions", []))
    merged["batch_format"] = existing.get("batch_format") or incoming.get("batch_format")
    merged["expiry_format"] = existing.get("expiry_format") or incoming.get("expiry_format")
    merged["manufacturing_date_format"] = existing.get("manufacturing_date_format") or incoming.get("manufacturing_date_format")
    merged["qr_format"] = existing.get("qr_format") or incoming.get("qr_format")
    merged["uv_required"] = bool(existing.get("uv_required", incoming.get("uv_required", True)))
    merged["product_status"] = existing.get("product_status") or incoming.get("product_status", "active")
    merged["source_url"] = incoming.get("source_url") or existing.get("source_url")
    merged["source_confidence"] = max(float(existing.get("source_confidence", 0.0)), float(incoming.get("source_confidence", 0.0)))

    source_list = list(existing.get("data_sources", [])) + list(incoming.get("data_sources", []))
    merged["data_sources"] = _compact_unique(source_list, limit=8)
    merged["last_refreshed"] = incoming.get("last_refreshed", int(time.time()))
    merged["last_verified"] = incoming.get("last_verified", int(time.time()))
    return merged


def _upsert_hash_state(state: Dict, digest: str, max_items: int = 5000) -> None:
    seen = state.setdefault("seen_hashes", {})
    seen[digest] = int(time.time())
    if len(seen) <= max_items:
        return

    # Keep most recent hashes only so state remains compact.
    ordered = sorted(seen.items(), key=lambda item: item[1], reverse=True)[:max_items]
    state["seen_hashes"] = {k: v for k, v in ordered}


def _record_snapshot(snapshots: List[Dict], source_id: str, status: str, message: str, records: int) -> None:
    snapshots.append(
        {
            "source_id": source_id,
            "status": status,
            "message": message,
            "records": int(records),
            "timestamp": int(time.time()),
        }
    )


def _prune_snapshots(snapshots: List[Dict], max_items: int = 150) -> List[Dict]:
    if len(snapshots) <= max_items:
        return snapshots
    return sorted(snapshots, key=lambda row: row.get("timestamp", 0), reverse=True)[:max_items]


def _feed_records(path: str) -> List[Dict]:
    data = _load_json(path, [])
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    return []


def _normalize_feed_row(raw: Dict, source_id: str, source_cfg: Dict) -> Optional[Tuple[str, Dict, str]]:
    brand = str(raw.get("brand") or raw.get("drug_name") or "").strip()
    if not brand:
        return None

    key = _normalize_key(brand)
    dosage = raw.get("dosage")
    if dosage is not None:
        try:
            dosage = int(dosage)
        except Exception:
            dosage = None

    normalized = {
        "product_id": str(raw.get("gtin") or raw.get("product_id") or key),
        "brand": brand,
        "generic_name": str(raw.get("generic_name") or brand),
        "dosage": dosage,
        "strength_unit": str(raw.get("strength_unit") or ("mg" if dosage else "")) or None,
        "dosage_form": str(raw.get("dosage_form") or "unknown"),
        "pack_presentation": raw.get("pack_presentation"),
        "manufacturer": str(raw.get("manufacturer") or "Unknown"),
        "manufacturer_license": raw.get("manufacturer_license"),
        "country": str(raw.get("country") or "IN"),
        "regulator": str(raw.get("regulator") or "CDSCO"),
        "expected_text_patterns": _compact_unique(list(raw.get("expected_text_patterns", [])) + [brand], limit=15),
        "aliases": _compact_unique(list(raw.get("aliases", [])) + [brand], limit=15),
        "known_ocr_distortions": list(raw.get("known_ocr_distortions", [])),
        "qr_format": str(raw.get("qr_format") or _build_qr_pattern(brand)),
        "batch_format": str(raw.get("batch_format") or "[A-Z0-9-]{4,20}"),
        "expiry_format": str(raw.get("expiry_format") or "(MM/YYYY|YYYY-MM)"),
        "manufacturing_date_format": str(raw.get("manufacturing_date_format") or "(MM/YYYY|YYYY-MM)"),
        "uv_required": bool(raw.get("uv_required", True)),
        "uv_signature": list(raw.get("uv_signature", [])),
        "reference_images": list(raw.get("reference_images", [])),
        "hologram_metadata": raw.get("hologram_metadata", {}),
        "layout_anchors": raw.get("layout_anchors", {}),
        "packaging_versions": list(raw.get("packaging_versions", [])),
        "product_status": str(raw.get("product_status") or "active"),
        "source_url": str(source_cfg.get("url") or source_id),
        "source_confidence": float(source_cfg.get("confidence", 0.9)),
        "data_sources": [source_id],
        "last_refreshed": int(time.time()),
        "last_verified": int(time.time()),
    }

    fingerprint = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return key, normalized, fingerprint


def _build_canonical_outputs(runtime_db: Dict[str, Dict], regulatory_cache: Dict[str, Dict]) -> Dict[str, List[Dict]]:
    products: List[Dict] = []
    packaging_profiles: List[Dict] = []
    regulatory_events: List[Dict] = []

    for key, entry in runtime_db.items():
        products.append(
            {
                "product_key": key,
                "product_id": entry.get("product_id") or key,
                "brand": entry.get("brand"),
                "generic_name": entry.get("generic_name"),
                "dosage": entry.get("dosage"),
                "strength_unit": entry.get("strength_unit"),
                "dosage_form": entry.get("dosage_form"),
                "pack_presentation": entry.get("pack_presentation"),
                "manufacturer": entry.get("manufacturer"),
                "manufacturer_license": entry.get("manufacturer_license"),
                "country": entry.get("country"),
                "regulator": entry.get("regulator"),
                "product_status": entry.get("product_status"),
                "aliases": entry.get("aliases", []),
                "expected_text_patterns": entry.get("expected_text_patterns", []),
                "batch_format": entry.get("batch_format"),
                "expiry_format": entry.get("expiry_format"),
                "manufacturing_date_format": entry.get("manufacturing_date_format"),
                "qr_format": entry.get("qr_format"),
                "source_url": entry.get("source_url"),
                "source_confidence": entry.get("source_confidence"),
                "last_verified": entry.get("last_verified"),
                "last_refreshed": entry.get("last_refreshed"),
            }
        )

        packaging_profiles.append(
            {
                "product_key": key,
                "reference_images": entry.get("reference_images", []),
                "uv_signature": entry.get("uv_signature", []),
                "uv_required": bool(entry.get("uv_required", True)),
                "hologram_metadata": entry.get("hologram_metadata", {}),
                "layout_anchors": entry.get("layout_anchors", {}),
                "packaging_versions": entry.get("packaging_versions", []),
            }
        )

    for key, row in regulatory_cache.items():
        for alert in row.get("active_alerts", []):
            regulatory_events.append(
                {
                    "product_key": key,
                    "source": row.get("source"),
                    "status": alert.get("status"),
                    "classification": alert.get("classification"),
                    "recalling_firm": alert.get("recalling_firm"),
                    "reason": alert.get("reason"),
                    "recall_initiation_date": alert.get("recall_initiation_date"),
                    "last_checked": row.get("last_checked"),
                }
            )

    return {
        "products": products,
        "packaging_profiles": packaging_profiles,
        "regulatory_events": regulatory_events,
    }


def _build_regulatory_cache(db: Dict[str, Dict], terms: List[str]) -> Dict[str, Dict]:
    source_cfg = _ensure_source_registry().get("openfda_enforcement", {})
    cache: Dict[str, Dict] = {}
    for term in terms:
        key = _normalize_key(term)
        if not bool(source_cfg.get("enabled", True)):
            cache[key] = {
                "checked": False,
                "source": "openfda_enforcement",
                "last_checked": int(time.time()),
                "error": "Source disabled in source_registry",
                "active_alerts": [],
            }
            continue
        try:
            recalls = _search_openfda_enforcement_with_config(term=term, source_cfg=source_cfg)
        except Exception as exc:
            cache[key] = {
                "checked": False,
                "source": "openfda_enforcement",
                "last_checked": int(time.time()),
                "error": str(exc),
                "active_alerts": [],
            }
            continue

        compact_alerts = []
        for record in recalls[:10]:
            compact_alerts.append(
                {
                    "status": record.get("status"),
                    "classification": record.get("classification"),
                    "recalling_firm": record.get("recalling_firm"),
                    "reason": record.get("reason_for_recall"),
                    "recall_initiation_date": record.get("recall_initiation_date"),
                }
            )

        cache[key] = {
            "checked": True,
            "source": "openfda_enforcement",
            "last_checked": int(time.time()),
            "active_alerts": compact_alerts,
        }

        if key in db:
            db[key]["regulatory_alert_count"] = len(compact_alerts)

    return cache


def update_runtime_database(
    base_db_path: str = "database/drug_db.json",
    runtime_db_path: str = "database/drug_db_runtime.json",
    state_path: str = "database/ingest_state.json",
    regulatory_cache_path: str = "database/regulatory_cache.json",
    per_term_limit: int = 8,
) -> Dict:
    registry = _ensure_source_registry()
    base_db = _load_json(base_db_path, {})
    state = _load_json(state_path, {"seen_hashes": {}, "last_run": None, "source_snapshots": []})

    runtime_db = dict(base_db)
    terms = [entry.get("brand", key) for key, entry in base_db.items()]
    terms = _compact_unique(terms, limit=200)

    fetched_records = 0
    inserted_or_updated = 0
    skipped_unchanged = 0
    fetch_errors = 0
    schema_errors: List[str] = []
    source_snapshots: List[Dict] = list(state.get("source_snapshots", []))

    label_cfg = registry.get("openfda_label", {})

    if bool(label_cfg.get("enabled", True)):
        for term in terms:
            try:
                records = _search_openfda_label_with_config(term=term, source_cfg=label_cfg, limit=per_term_limit)
                _record_snapshot(source_snapshots, "openfda_label", "ok", f"Fetched for term '{term}'", len(records))
            except Exception as exc:
                fetch_errors += 1
                _record_snapshot(source_snapshots, "openfda_label", "error", str(exc), 0)
                continue

            for raw in records:
                normalized_bundle = _normalize_label_record(raw)
                if not normalized_bundle:
                    continue

                fetched_records += 1
                key, normalized, fingerprint = normalized_bundle
                normalized["source_url"] = label_cfg.get("url", OPENFDA_LABEL_ENDPOINT)
                normalized["source_confidence"] = float(label_cfg.get("confidence", 0.9))

                digest = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()
                if digest in state.get("seen_hashes", {}):
                    skipped_unchanged += 1
                    continue

                runtime_db[key] = _merge_entry(runtime_db.get(key, {}), normalized)
                _upsert_hash_state(state, digest)
                inserted_or_updated += 1
    else:
        _record_snapshot(source_snapshots, "openfda_label", "disabled", "Source disabled in source_registry", 0)

    for source_id in ["india_cdsco_bulletin", "manufacturer_feed", "distributor_feed"]:
        source_cfg = registry.get(source_id, {})
        if not bool(source_cfg.get("enabled", False)):
            _record_snapshot(source_snapshots, source_id, "disabled", "Source disabled in source_registry", 0)
            continue

        path = str(source_cfg.get("url", "")).strip()
        rows = _feed_records(path)
        if not rows:
            _record_snapshot(source_snapshots, source_id, "ok", "No rows found in feed", 0)
            continue

        _record_snapshot(source_snapshots, source_id, "ok", f"Loaded {len(rows)} rows from feed", len(rows))
        for raw in rows:
            normalized_bundle = _normalize_feed_row(raw, source_id=source_id, source_cfg=source_cfg)
            if not normalized_bundle:
                continue

            fetched_records += 1
            key, normalized, fingerprint = normalized_bundle
            digest = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()
            if digest in state.get("seen_hashes", {}):
                skipped_unchanged += 1
                continue

            runtime_db[key] = _merge_entry(runtime_db.get(key, {}), normalized)
            _upsert_hash_state(state, digest)
            inserted_or_updated += 1

    # Enrich manual curated base records with canonical fields expected by full schema.
    for key, entry in list(runtime_db.items()):
        if "product_id" not in entry:
            entry["product_id"] = key
        if "generic_name" not in entry:
            entry["generic_name"] = entry.get("brand") or key
        if "strength_unit" not in entry:
            entry["strength_unit"] = "mg" if isinstance(entry.get("dosage"), int) else None
        entry.setdefault("dosage_form", "unknown")
        entry.setdefault("pack_presentation", None)
        entry.setdefault("manufacturer_license", None)
        entry.setdefault("country", "unknown")
        entry.setdefault("regulator", "unknown")
        entry.setdefault("batch_format", "[A-Z0-9-]{4,20}")
        entry.setdefault("expiry_format", "(MM/YYYY|YYYY-MM)")
        entry.setdefault("manufacturing_date_format", "(MM/YYYY|YYYY-MM)")
        entry.setdefault("hologram_metadata", {})
        entry.setdefault("layout_anchors", {})
        entry.setdefault("packaging_versions", [])
        entry.setdefault("product_status", "active")
        entry.setdefault("source_url", "database/drug_db.json")
        entry.setdefault("source_confidence", 0.98)
        entry.setdefault("last_verified", int(time.time()))
        entry.setdefault("last_refreshed", int(time.time()))

        schema_errors.extend(_schema_errors(key, entry))

    regulatory_cache = _build_regulatory_cache(runtime_db, terms)
    canonical = _build_canonical_outputs(runtime_db, regulatory_cache)

    state["last_run"] = int(time.time())
    state["source_snapshots"] = _prune_snapshots(source_snapshots, max_items=150)
    _save_json(runtime_db_path, runtime_db)
    _save_json(state_path, state)
    _save_json(regulatory_cache_path, regulatory_cache)
    _save_json("database/canonical/products.json", canonical["products"])
    _save_json("database/canonical/packaging_profiles.json", canonical["packaging_profiles"])
    _save_json("database/canonical/regulatory_events.json", canonical["regulatory_events"])
    _save_json("database/canonical/source_snapshots.json", state["source_snapshots"])
    _save_json(
        "database/canonical/quality_report.json",
        {
            "generated_at": int(time.time()),
            "schema_error_count": len(schema_errors),
            "schema_errors": schema_errors[:500],
            "freshness_sla": {
                "regulatory_refresh": "weekly",
                "product_master_refresh": "monthly",
            },
        },
    )

    return {
        "ok": True,
        "base_db_entries": len(base_db),
        "runtime_db_entries": len(runtime_db),
        "fetched_records": fetched_records,
        "inserted_or_updated": inserted_or_updated,
        "skipped_unchanged": skipped_unchanged,
        "fetch_errors": fetch_errors,
        "schema_error_count": len(schema_errors),
        "state_hash_count": len(state.get("seen_hashes", {})),
        "regulatory_cache_entries": len(regulatory_cache),
        "canonical_products": len(canonical["products"]),
        "canonical_packaging_profiles": len(canonical["packaging_profiles"]),
        "canonical_regulatory_events": len(canonical["regulatory_events"]),
        "runtime_db_path": runtime_db_path,
        "regulatory_cache_path": regulatory_cache_path,
    }


def main():
    parser = argparse.ArgumentParser(description="Update runtime pharmacy database from external sources")
    parser.add_argument("--base", default="database/drug_db.json", help="Path to base curated DB")
    parser.add_argument("--runtime", default="database/drug_db_runtime.json", help="Path to generated runtime DB")
    parser.add_argument("--state", default="database/ingest_state.json", help="Path to ingest state file")
    parser.add_argument("--regulatory-cache", default="database/regulatory_cache.json", help="Path to regulatory cache")
    parser.add_argument("--limit", type=int, default=8, help="Records per source query term")
    args = parser.parse_args()

    summary = update_runtime_database(
        base_db_path=args.base,
        runtime_db_path=args.runtime,
        state_path=args.state,
        regulatory_cache_path=args.regulatory_cache,
        per_term_limit=max(args.limit, 1),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
