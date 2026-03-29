import json
import os
import argparse
from typing import Dict, Optional

import cv2

from src.decision.candidate_generator import get_best_drug_candidates
from src.decision.decision_engine import verify
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


def load_database(path: str = "database/drug_db.json") -> Dict:
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

    # Final decision
    decision = verify(
        candidates=candidates,
        database=database,
        text=ocr_result["normalized_text"],
        ocr_confidence=ocr_result["confidence"],
        qr_result=qr_result,
        uv_result=uv_result,
    )

    return {
        "ocr": ocr_result,
        "candidates": candidates,
        "qr": qr_result,
        "uv": uv_result,
        "decision": decision,
    }


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