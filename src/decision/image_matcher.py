from typing import Dict, List, Optional

import cv2
import numpy as np


def _safe_gray(image: np.ndarray) -> np.ndarray:
    if image is None:
        return None
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _orb_similarity(img_a: np.ndarray, img_b: np.ndarray) -> float:
    if img_a is None or img_b is None:
        return 0.0

    orb = cv2.ORB_create(nfeatures=800)
    kp_a, desc_a = orb.detectAndCompute(_safe_gray(img_a), None)
    kp_b, desc_b = orb.detectAndCompute(_safe_gray(img_b), None)

    if desc_a is None or desc_b is None or len(desc_a) == 0 or len(desc_b) == 0:
        return 0.0

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(desc_a, desc_b)
    if not matches:
        return 0.0

    distances = [m.distance for m in matches]
    mean_distance = float(np.mean(distances))
    return max(0.0, 1.0 - min(mean_distance / 100.0, 1.0))


def match_against_references(query_image: np.ndarray, reference_paths: List[str]) -> Dict:
    if query_image is None:
        return {
            "available": False,
            "score": 0.0,
            "best_reference": None,
            "reason": "No normal image provided",
        }

    if not reference_paths:
        return {
            "available": False,
            "score": 0.0,
            "best_reference": None,
            "reason": "No dataset references configured",
        }

    best_score = 0.0
    best_ref = None
    for path in reference_paths:
        ref_img = cv2.imread(path)
        if ref_img is None:
            continue
        score = _orb_similarity(query_image, ref_img)
        if score > best_score:
            best_score = score
            best_ref = path

    return {
        "available": best_ref is not None,
        "score": float(best_score),
        "best_reference": best_ref,
        "reason": "Compared against known legal packaging dataset",
    }
