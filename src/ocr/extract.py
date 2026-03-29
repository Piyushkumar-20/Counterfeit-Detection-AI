from typing import Any, Dict, List, Tuple, Union
import cv2
import importlib
import numpy as np
import pytesseract

from src.ocr.normalizer import normalize_ocr_errors
from src.preprocessing.align import perspective_correct
from src.preprocessing.enhance import build_ocr_variants

EASYOCR_AVAILABLE = importlib.util.find_spec("easyocr") is not None
_EASYOCR_READER = None


ImageInput = Union[str, np.ndarray]


def _to_image(image_input: ImageInput) -> np.ndarray:
    if isinstance(image_input, str):
        image = cv2.imread(image_input)
        if image is None:
            raise ValueError(f"Image not found: {image_input}")
        return image

    if image_input is None:
        raise ValueError("Input image is None")

    return image_input


def _extract_with_tesseract(image: np.ndarray, psm: int = 6) -> str:
    config = (
        f"--oem 3 --psm {psm} "
        "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-/."
    )
    return pytesseract.image_to_string(image, config=config)


def _extract_with_easyocr(image: np.ndarray) -> str:
    if not EASYOCR_AVAILABLE:
        return ""

    global _EASYOCR_READER
    if _EASYOCR_READER is None:
        easyocr_module = importlib.import_module("easyocr")
        _EASYOCR_READER = easyocr_module.Reader(["en"], gpu=False)

    result = _EASYOCR_READER.readtext(image, detail=0, paragraph=True)
    return " ".join(result).strip()


def _score_text(text: str) -> float:
    if not text:
        return 0.0

    alnum = sum(ch.isalnum() for ch in text)
    length = max(len(text), 1)
    ratio = alnum / length
    return min(max(ratio, 0.0), 1.0)


def _dedupe_keep_order(values: List[str]) -> List[str]:
    seen = set()
    out = []
    for value in values:
        key = value.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(value)
    return out


def extract_text(image_input: ImageInput, debug: bool = False) -> Dict[str, Any]:
    """
    Multi-pass OCR that combines several preprocessed variants and OCR engines.

    Returns a structured object:
    {
      "raw_text": str,
      "cleaned_text": str,
      "normalized_text": str,
      "confidence": float,
      "passes": [{"engine": str, "variant": str, "text": str, "score": float}]
    }
    """
    original = _to_image(image_input)
    aligned = perspective_correct(original)

    variants = build_ocr_variants(aligned)
    passes: List[Dict[str, Any]] = []

    for variant_name, variant_image in variants:
        tess_text = _extract_with_tesseract(variant_image, psm=6)
        tess_score = _score_text(tess_text)
        passes.append(
            {
                "engine": "tesseract",
                "variant": variant_name,
                "text": tess_text,
                "score": tess_score,
            }
        )

        if EASYOCR_AVAILABLE:
            easy_text = _extract_with_easyocr(variant_image)
            easy_score = _score_text(easy_text)
            passes.append(
                {
                    "engine": "easyocr",
                    "variant": variant_name,
                    "text": easy_text,
                    "score": easy_score,
                }
            )

    if not passes:
        return {
            "raw_text": "",
            "cleaned_text": "",
            "normalized_text": "",
            "confidence": 0.0,
            "passes": [],
        }

    ordered_passes = sorted(passes, key=lambda row: row["score"], reverse=True)
    top_passes = ordered_passes[:3]
    merged_raw = "\n".join(
        _dedupe_keep_order([entry["text"] for entry in top_passes if entry["text"]])
    )

    cleaned_text = " ".join(merged_raw.split())
    normalized_text = normalize_ocr_errors(cleaned_text)
    confidence = float(np.mean([entry["score"] for entry in top_passes])) if top_passes else 0.0

    if debug:
        print("\n--- OCR TOP PASSES ---")
        for row in top_passes:
            print(f"{row['engine']}:{row['variant']} => {row['score']:.3f}")

    return {
        "raw_text": merged_raw,
        "cleaned_text": cleaned_text,
        "normalized_text": normalized_text,
        "confidence": confidence,
        "passes": ordered_passes,
    }


def extract_text_legacy(image_input: ImageInput, debug: bool = False) -> Tuple[str, str]:
    """Backwards-compatible 2-value output for older call sites."""
    result = extract_text(image_input=image_input, debug=debug)
    return result["raw_text"], result["normalized_text"]