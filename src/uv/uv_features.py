from typing import Dict, List
import cv2
import numpy as np


def _ensure_gray(image: np.ndarray) -> np.ndarray:
	if len(image.shape) == 2:
		return image
	return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def extract_uv_signature(image: np.ndarray) -> Dict:
	gray = _ensure_gray(image)

	orb = cv2.ORB_create(nfeatures=600)
	keypoints, descriptors = orb.detectAndCompute(gray, None)

	intensity_mean = float(np.mean(gray)) / 255.0
	intensity_std = float(np.std(gray)) / 255.0

	return {
		"keypoint_count": len(keypoints) if keypoints else 0,
		"descriptors": descriptors,
		"intensity_mean": intensity_mean,
		"intensity_std": intensity_std,
	}


def _descriptor_similarity(desc_a: np.ndarray, desc_b: np.ndarray) -> float:
	if desc_a is None or desc_b is None or len(desc_a) == 0 or len(desc_b) == 0:
		return 0.0

	matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
	matches = matcher.match(desc_a, desc_b)

	if not matches:
		return 0.0

	distances = [m.distance for m in matches]
	avg_distance = float(np.mean(distances))
	return 1.0 - min(avg_distance / 100.0, 1.0)


def compare_uv_signatures(query: Dict, references: List[Dict]) -> float:
	if not references:
		return 0.0

	scores = []
	for reference in references:
		feature_score = _descriptor_similarity(query.get("descriptors"), reference.get("descriptors"))
		intensity_gap = abs(float(query.get("intensity_mean", 0.0)) - float(reference.get("intensity_mean", 0.0)))
		intensity_score = max(0.0, 1.0 - intensity_gap)
		combined = (feature_score * 0.8) + (intensity_score * 0.2)
		scores.append(combined)

	return float(max(scores)) if scores else 0.0
