import json
import os
import argparse
from typing import Dict, Optional

import cv2
import numpy as np

from src.decision.candidate_generator import get_best_drug_candidates
from src.decision.decision_engine import verify
from src.decision.image_matcher import match_against_references
from src.decision.regulatory_sources import verify_with_regulatory_sources
from src.ocr.extract import extract_text
from src.qrcode.decoder import QRDecoder
from src.qrcode.detector import QRDetector
from src.uv.uv_detector import UVDetector


def _load_image(path: Optional[str]):
    if not path:
        return None
    if not os.path.exists(path):
        return None
    return cv2.imread(path)


def load_database(path: str = "database/drug_db_runtime.json") -> Dict:
    if not os.path.exists(path):
        path = "database/drug_db.json"
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _parse_uv_reference_images(entry: Dict):
    refs = []
    for rel_path in entry.get("uv_signature", []):
        image = _load_image(rel_path)
        if image is not None:
            refs.append(image)
    return refs


def process_image(normal_image_path: str, uv_image_path: Optional[str] = None, debug: bool = False) -> Dict:
    database = load_database()

    normal_img = _load_image(normal_image_path)
    if normal_img is None:
        raise ValueError(f"Normal image not found: {normal_image_path}")

    uv_img = _load_image(uv_image_path)

    # OCR pipeline
    ocr_result = extract_text(normal_img, debug=debug)

    # Candidate generation
    candidates = get_best_drug_candidates(
        text=ocr_result["normalized_text"],
        database=database,
        k=5,
    )
    top_entry = database.get(candidates[0]["name"], {}) if candidates else {}

    # QR pipeline
    qr_detector = QRDetector()
    qr_decoder = QRDecoder()
    detection = qr_detector.detect(normal_img)

    qr_result = {
        "found": False,
        "decoded": False,
        "data": None,
        "format_score": 0.0,
        "reason": "QR not detected",
    }

    if detection["found"] and detection["cropped"] is not None:
        decoded = qr_decoder.decode(detection["cropped"])
        if decoded["success"]:
            qr_validation = qr_decoder.validate(
                decoded["data"],
                expected_pattern=top_entry.get("qr_format"),
            )
            qr_result = {
                "found": True,
                "decoded": True,
                "data": decoded["data"],
                "method": decoded.get("method"),
                "format_score": qr_validation["format_score"],
                "is_structured": qr_validation["is_structured"],
                "signature_valid": qr_validation["signature_valid"],
                "reason": qr_validation["reason"],
            }
        else:
            qr_result = {
                "found": True,
                "decoded": False,
                "data": None,
                "format_score": 0.1,
                "reason": "QR detected but not decodable",
            }

    # UV pipeline
    uv_detector = UVDetector()
    uv_references = _parse_uv_reference_images(top_entry)
    uv_result = uv_detector.analyze(uv_img, reference_images=uv_references)

    # Dataset image matching (known legal package references)
    image_match_result = match_against_references(
        query_image=normal_img,
        reference_paths=top_entry.get("reference_images", []),
    )

    # Regulatory source verification (optional online APIs)
    regulatory_sources_result = verify_with_regulatory_sources(
        drug_name=candidates[0]["name"] if candidates else None,
        dosage=top_entry.get("dosage") if top_entry else None,
        qr_data=qr_result.get("data"),
    )

    # Final decision
    decision = verify(
        candidates=candidates,
        database=database,
        text=ocr_result["normalized_text"],
        ocr_confidence=ocr_result["confidence"],
        qr_result=qr_result,
        uv_result=uv_result,
        image_match_score=image_match_result.get("score", 0.5),
        regulatory_sources_result=regulatory_sources_result,
    )

    return {
        "ocr": ocr_result,
        "candidates": candidates,
        "qr": qr_result,
        "uv": uv_result,
        "image_dataset_match": image_match_result,
        "regulatory_sources": regulatory_sources_result,
        "decision": decision,
    }


def process_captured_images(normal_image: np.ndarray, uv_image: Optional[np.ndarray] = None, debug: bool = False) -> Dict:
    """
    Hardware-friendly entry point: pass captured normal and UV frames directly.
    Uses unique temp files to support parallel capture requests safely.
    """
    if normal_image is None:
        raise ValueError("normal_image cannot be None")

    import tempfile

    os.makedirs("data/processed", exist_ok=True)
    normal_tmp = tempfile.NamedTemporaryFile(prefix="runtime_normal_", suffix=".png", dir="data/processed", delete=False)
    uv_tmp = None
    try:
        temp_normal = normal_tmp.name
        cv2.imwrite(temp_normal, normal_image)
        uv_path = None
        if uv_image is not None:
            uv_tmp = tempfile.NamedTemporaryFile(prefix="runtime_uv_", suffix=".png", dir="data/processed", delete=False)
            uv_path = uv_tmp.name
            cv2.imwrite(uv_path, uv_image)

        return process_image(normal_image_path=temp_normal, uv_image_path=uv_path, debug=debug)
    finally:
        normal_tmp.close()
        if uv_tmp is not None:
            uv_tmp.close()
        if os.path.exists(normal_tmp.name):
            os.remove(normal_tmp.name)
        if uv_tmp is not None and os.path.exists(uv_tmp.name):
            os.remove(uv_tmp.name)


def main():
    parser = argparse.ArgumentParser(description="Pharmacy AI image verifier")
    parser.add_argument("--normal", dest="normal_path", default=None, help="Path to normal-light package image")
    parser.add_argument("--uv", dest="uv_path", default=None, help="Optional path to UV image")
    parser.add_argument("--debug", action="store_true", help="Enable OCR debug logging")
    args = parser.parse_args()

    normal_candidates = [
        args.normal_path,
        "data/raw/normal/sample.png",
        "data/raw/normal/sample.jpg",
        "data/raw/normal/sample.jpeg",
    ]
    sample_path = next((path for path in normal_candidates if path and os.path.exists(path)), None)

    if sample_path is None:
        print("No normal image found. Use --normal <path> or place an image in data/raw/normal/.")
        return

    uv_candidates = [
        args.uv_path,
        "data/raw/uv/sample.png",
        "data/raw/uv/sample.jpg",
        "data/raw/uv/sample.jpeg",
    ]
    uv_path = next((path for path in uv_candidates if path and os.path.exists(path)), None)

    result = process_image(sample_path, uv_image_path=uv_path, debug=args.debug)

    print("\n====== FINAL RESULT ======")
    print(json.dumps(result["decision"], indent=2))


if __name__ == "__main__":
    main()