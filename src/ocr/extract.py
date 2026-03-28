import pytesseract
import cv2
import re


def preprocess_image(image, debug=False):
    image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    _, thresh = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    thresh = cv2.bitwise_not(thresh)

    if debug:
        cv2.imwrite("debug.png", thresh)

    return thresh


def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s\-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_text(image_path, debug=False):
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Image not found: {image_path}")

    processed = preprocess_image(image, debug)

    config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-'

    raw_text = pytesseract.image_to_string(processed, config=config)

    cleaned_text = clean_text(raw_text)

    if debug:
        print("\n--- RAW OCR ---\n", raw_text)
        print("\n--- CLEANED OCR ---\n", cleaned_text)

    return raw_text, cleaned_text