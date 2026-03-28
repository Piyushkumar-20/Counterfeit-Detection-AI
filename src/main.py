import cv2
import os

# OCR
from src.ocr.extract import extract_text

# Decision
from src.decision.decision_engine import make_decision

# QR
from src.qrcode.detector import QRDetector
from src.qrcode.decoder import QRDecoder


def load_image(path):
    if not os.path.exists(path):
        return None
    return cv2.imread(path)


def process_image(normal_img):

    # -------- OCR --------
    raw_text = extract_text(normal_img)

    # -------- QR --------
    qr_detector = QRDetector()
    qr_decoder = QRDecoder()

    qr_detection = qr_detector.detect(normal_img)

    qr_result = {
        "found": False,
        "data": None
    }

    if qr_detection["found"]:
        qr_decode = qr_decoder.decode(qr_detection["cropped"])

        if qr_decode["success"]:
            qr_result = {
                "found": True,
                "data": qr_decode["data"]
            }

    # -------- DECISION --------
    decision = make_decision(
        text=raw_text,
        qr_result=qr_result
    )

    return {
        "text": raw_text,
        "qr": qr_result,
        "decision": decision
    }


def main():
    normal_path = "data/raw/normal/sample.jpg"  # adjust if needed

    normal_img = load_image(normal_path)

    if normal_img is None:
        print("Image not found")
        return

    result = process_image(normal_img)

    print("\n====== OCR TEXT ======")
    print(result["text"])

    print("\n====== QR RESULT ======")
    print(result["qr"])

    print("\n====== FINAL DECISION ======")
    print(result["decision"])


if __name__ == "__main__":
    main()