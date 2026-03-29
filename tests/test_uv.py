import numpy as np

from src.uv.uv_detector import UVDetector


def test_uv_detector_handles_missing_uv_image():
	detector = UVDetector()
	result = detector.analyze(None, reference_images=[])
	assert result["available"] is False
	assert result["similarity"] == 0.0


def test_uv_detector_produces_similarity_score():
	detector = UVDetector()
	image = np.zeros((128, 128, 3), dtype=np.uint8)
	image[32:96, 32:96] = 255

	result = detector.analyze(image, reference_images=[image])
	assert result["available"] is True
	assert 0.0 <= result["similarity"] <= 1.0
