from typing import List, Tuple
import cv2
import numpy as np


def _ensure_gray(image: np.ndarray) -> np.ndarray:
	if len(image.shape) == 2:
		return image
	return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def enhance_for_ocr(image: np.ndarray) -> np.ndarray:
	gray = _ensure_gray(image)
	clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
	contrast = clahe.apply(gray)
	denoised = cv2.fastNlMeansDenoising(contrast, None, 10, 7, 21)
	sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
	sharpened = cv2.filter2D(denoised, -1, sharpen_kernel)
	return sharpened


def build_ocr_variants(image: np.ndarray) -> List[Tuple[str, np.ndarray]]:
	"""
	Build multiple threshold variants to improve OCR robustness under noise.
	"""
	enhanced = enhance_for_ocr(image)

	_, otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	adaptive = cv2.adaptiveThreshold(
		enhanced,
		255,
		cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
		cv2.THRESH_BINARY,
		17,
		7,
	)
	inverted = cv2.bitwise_not(otsu)

	return [
		("enhanced", enhanced),
		("otsu", otsu),
		("adaptive", adaptive),
		("inverted", inverted),
	]
