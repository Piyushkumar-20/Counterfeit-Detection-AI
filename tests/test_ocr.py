from src.ocr.extractors import extract_dosage
from src.ocr.normalizer import normalize_ocr_errors


def test_normalize_ocr_errors_recovers_common_500mg_pattern():
	raw = "PARACETAMOL 5OO mg"
	normalized = normalize_ocr_errors(raw)
	assert "500" in normalized


def test_extract_dosage_reads_mg_value():
	text = "crocin 500 mg tablet"
	assert extract_dosage(text) == 500
