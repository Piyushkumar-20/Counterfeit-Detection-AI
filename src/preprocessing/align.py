from typing import Tuple
import cv2
import numpy as np


def _order_points(points: np.ndarray) -> np.ndarray:
	rect = np.zeros((4, 2), dtype="float32")
	s = points.sum(axis=1)
	rect[0] = points[np.argmin(s)]
	rect[2] = points[np.argmax(s)]

	diff = np.diff(points, axis=1)
	rect[1] = points[np.argmin(diff)]
	rect[3] = points[np.argmax(diff)]
	return rect


def _safe_warp(image: np.ndarray, contour: np.ndarray) -> np.ndarray:
	rect = _order_points(contour.reshape(4, 2))
	(tl, tr, br, bl) = rect

	width_a = np.linalg.norm(br - bl)
	width_b = np.linalg.norm(tr - tl)
	max_width = int(max(width_a, width_b))

	height_a = np.linalg.norm(tr - br)
	height_b = np.linalg.norm(tl - bl)
	max_height = int(max(height_a, height_b))

	if max_width < 50 or max_height < 50:
		return image

	destination = np.array(
		[[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
		dtype="float32",
	)
	matrix = cv2.getPerspectiveTransform(rect, destination)
	return cv2.warpPerspective(image, matrix, (max_width, max_height))


def perspective_correct(image: np.ndarray) -> np.ndarray:
	"""
	Best-effort perspective correction for angled/curved captures.
	Falls back to original image when no suitable quadrilateral is found.
	"""
	if image is None or image.size == 0:
		return image

	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (5, 5), 0)
	edged = cv2.Canny(blurred, 40, 140)

	contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	contours = sorted(contours, key=cv2.contourArea, reverse=True)

	for contour in contours[:12]:
		perimeter = cv2.arcLength(contour, True)
		approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
		if len(approx) == 4:
			return _safe_warp(image, approx)

	return image
