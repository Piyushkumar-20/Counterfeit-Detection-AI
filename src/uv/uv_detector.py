from typing import Dict, List, Optional
import cv2

from src.uv.uv_features import compare_uv_signatures, extract_uv_signature


class UVDetector:
	def __init__(self, similarity_threshold: float = 0.55):
		self.similarity_threshold = similarity_threshold

	def analyze(self, uv_image, reference_images: Optional[List] = None) -> Dict:
		if uv_image is None:
			return {
				"available": False,
				"similarity": 0.0,
				"uv_present": False,
				"reason": "No UV image provided",
			}

		query_sig = extract_uv_signature(uv_image)

		references = []
		if reference_images:
			for image in reference_images:
				if image is None:
					continue
				references.append(extract_uv_signature(image))

		similarity = compare_uv_signatures(query_sig, references)
		uv_present = similarity >= self.similarity_threshold if references else query_sig["keypoint_count"] > 80

		return {
			"available": True,
			"similarity": float(similarity),
			"uv_present": bool(uv_present),
			"keypoint_count": int(query_sig["keypoint_count"]),
			"reason": "Compared against reference UV templates" if references else "Reference UV templates unavailable",
		}
