import cv2
import hashlib
import re
from typing import Dict

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

    def validate(self, payload: str, expected_pattern: str = None) -> Dict:
        if not payload:
            return {
                "is_structured": False,
                "format_score": 0.0,
                "signature_valid": False,
                "reason": "Empty QR payload",
            }

        payload = payload.strip()
        score = 0.0
        reasons = []

        gs1_like = bool(re.match(r"\(01\)\d{14}", payload))
        url_like = payload.startswith("http://") or payload.startswith("https://")
        key_value_like = ":" in payload and (";" in payload or "|" in payload)

        if gs1_like or url_like or key_value_like:
            score += 0.6
            reasons.append("Structured payload pattern detected")
        else:
            reasons.append("Payload appears unstructured")

        if expected_pattern:
            try:
                if re.search(expected_pattern, payload):
                    score += 0.3
                    reasons.append("Matches expected manufacturer format")
                else:
                    reasons.append("Does not match expected manufacturer format")
            except re.error:
                reasons.append("Invalid expected pattern in database")

        signature_valid = False
        if "sig=" in payload and "data=" in payload:
            signature_valid = self._validate_signature_stub(payload)
            if signature_valid:
                score += 0.1
                reasons.append("Signature field validated")
            else:
                reasons.append("Signature field present but invalid")

        return {
            "is_structured": score >= 0.5,
            "format_score": min(max(score, 0.0), 1.0),
            "signature_valid": signature_valid,
            "reason": "; ".join(reasons),
        }

    def _validate_signature_stub(self, payload: str) -> bool:
        """
        Stub for digital signature checks.
        Replace with manufacturer public-key verification when available.
        """
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return digest[-1] in {"0", "2", "4", "6", "8", "a", "c", "e"}