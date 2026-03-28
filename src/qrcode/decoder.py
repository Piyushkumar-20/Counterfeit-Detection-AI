# src/qr/decoder.py

import cv2

try:
    from pyzbar.pyzbar import decode as zbar_decode
    PYZBAR_AVAILABLE = True
except:
    PYZBAR_AVAILABLE = False


class QRDecoder:
    def __init__(self):
        self.detector = cv2.QRCodeDetector()

    def decode_opencv(self, image):
        """
        Try decoding using OpenCV
        """
        data, points, _ = self.detector.detectAndDecode(image)

        if data and data.strip():
            return data.strip()

        return None

    def decode_pyzbar(self, image):
        """
        Fallback decoding using pyzbar
        """
        if not PYZBAR_AVAILABLE:
            return None

        decoded = zbar_decode(image)

        if decoded:
            return decoded[0].data.decode("utf-8")

        return None

    def decode(self, image):
        """
        Main decode function with fallback
        """

        # Try OpenCV first
        data = self.decode_opencv(image)
        if data:
            return {
                "success": True,
                "data": data,
                "method": "opencv"
            }

        # Fallback to pyzbar
        data = self.decode_pyzbar(image)
        if data:
            return {
                "success": True,
                "data": data,
                "method": "pyzbar"
            }

        return {
            "success": False,
            "data": None,
            "method": None
        }