# src/qr/detector.py

import cv2
import numpy as np


class QRDetector:
    def __init__(self):
        self.detector = cv2.QRCodeDetector()

    def preprocess(self, image):
        """
        Preprocess image to improve detection robustness
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Improve contrast
        equalized = cv2.equalizeHist(blurred)

        return equalized

    def detect(self, image):
        """
        Detect QR code in image

        Returns:
            dict:
            {
                "found": bool,
                "bbox": np.array or None,
                "cropped": image or None
            }
        """

        processed = self.preprocess(image)

        found, points = self.detector.detect(processed)

        if not found or points is None:
            return {
                "found": False,
                "bbox": None,
                "cropped": None
            }

        points = points[0].astype(int)

        x, y, w, h = cv2.boundingRect(points)
        cropped = image[y:y + h, x:x + w]

        return {
            "found": True,
            "bbox": points,
            "cropped": cropped
        }