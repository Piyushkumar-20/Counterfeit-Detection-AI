from src.data_ingestion.updater import (
    _build_canonical_outputs,
    _merge_entry,
    _normalize_feed_row,
    _normalize_label_record,
    _schema_errors,
)


def test_normalize_label_record_extracts_core_fields():
    raw = {
        "openfda": {
            "brand_name": ["Crocin"],
            "generic_name": ["Paracetamol"],
            "manufacturer_name": ["GSK"],
        },
        "active_ingredient": ["Paracetamol 500 mg"],
    }
    row = _normalize_label_record(raw)
    assert row is not None
    key, normalized, fingerprint = row

    assert key == "crocin"
    assert normalized["brand"] == "Crocin"
    assert normalized["dosage"] == 500
    assert "openfda_label" in normalized["data_sources"]
    assert isinstance(fingerprint, str) and len(fingerprint) > 10


def test_merge_entry_keeps_existing_dosage_and_extends_aliases():
    existing = {
        "brand": "Crocin",
        "dosage": 500,
        "aliases": ["Crocin 500"],
        "expected_text_patterns": ["crocin", "500 mg"],
        "uv_signature": ["data/ref/uv/crocin.png"],
        "reference_images": ["data/ref/normal/crocin.png"],
        "data_sources": ["manual"],
    }
    incoming = {
        "brand": "Crocin",
        "dosage": 650,
        "manufacturer": "GSK",
        "aliases": ["Paracetamol"],
        "expected_text_patterns": ["paracetamol"],
        "qr_format": "(CROCIN|GTIN)",
        "uv_required": True,
        "data_sources": ["openfda_label"],
    }

    merged = _merge_entry(existing, incoming)
    assert merged["dosage"] == 500
    assert "Crocin 500" in merged["aliases"]
    assert "Paracetamol" in merged["aliases"]
    assert "manual" in merged["data_sources"]
    assert "openfda_label" in merged["data_sources"]
    assert merged["uv_signature"] == ["data/ref/uv/crocin.png"]


def test_schema_errors_flags_missing_required_fields():
    bad = {
        "brand": "",
        "manufacturer": "",
        "aliases": [],
        "expected_text_patterns": [],
    }
    errors = _schema_errors("sample", bad)
    assert len(errors) >= 4


def test_build_canonical_outputs_shapes_tables():
    runtime = {
        "crocin": {
            "product_id": "890000000001",
            "brand": "Crocin",
            "generic_name": "Paracetamol",
            "dosage": 500,
            "strength_unit": "mg",
            "dosage_form": "tablet",
            "pack_presentation": "strip-10",
            "manufacturer": "GSK",
            "manufacturer_license": "ABC123",
            "country": "IN",
            "regulator": "CDSCO",
            "product_status": "active",
            "aliases": ["Crocin 500"],
            "expected_text_patterns": ["crocin", "500 mg"],
            "batch_format": "[A-Z0-9-]{4,20}",
            "expiry_format": "MM/YYYY",
            "manufacturing_date_format": "MM/YYYY",
            "qr_format": "(CROCIN|GTIN)",
            "source_url": "https://example.com",
            "source_confidence": 0.95,
            "last_verified": 1,
            "last_refreshed": 1,
            "reference_images": ["x.png"],
            "uv_signature": ["u.png"],
            "uv_required": True,
            "hologram_metadata": {},
            "layout_anchors": {},
            "packaging_versions": [],
        }
    }
    reg = {
        "crocin": {
            "source": "openfda_enforcement",
            "last_checked": 2,
            "active_alerts": [
                {
                    "status": "ongoing",
                    "classification": "Class II",
                    "recalling_firm": "Firm",
                    "reason": "Reason",
                    "recall_initiation_date": "20240101",
                }
            ],
        }
    }

    out = _build_canonical_outputs(runtime, reg)
    assert len(out["products"]) == 1
    assert len(out["packaging_profiles"]) == 1
    assert len(out["regulatory_events"]) == 1


def test_normalize_feed_row_parses_gtin_and_brand():
    raw = {
        "gtin": "890000000001",
        "brand": "Crocin",
        "dosage": 500,
        "manufacturer": "GSK",
        "expected_text_patterns": ["crocin"],
        "aliases": ["Crocin 500"],
    }
    cfg = {"url": "database/feeds/manufacturer_feed.json", "confidence": 0.95}
    parsed = _normalize_feed_row(raw, source_id="manufacturer_feed", source_cfg=cfg)
    assert parsed is not None
    key, normalized, _ = parsed
    assert key == "crocin"
    assert normalized["product_id"] == "890000000001"
    assert normalized["brand"] == "Crocin"
