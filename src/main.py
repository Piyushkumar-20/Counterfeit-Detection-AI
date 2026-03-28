import cv2
import pytesseract
import numpy as np
import re
import json
import os
from difflib import get_close_matches

# -----------------------------
# CONFIG
# -----------------------------

KNOWN_DRUGS = ["paracetamol", "crocin", "dolo", "paracip"]
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

NORMAL_IMAGE_PATH = "data/raw/normal/sample.png"
UV_IMAGE_PATH = "data/raw/uv/sample.png"
DB_PATH = "../database/drug_db.json"

print("Normal exists:", os.path.exists(NORMAL_IMAGE_PATH))
print("UV exists:", os.path.exists(UV_IMAGE_PATH))


# -----------------------------
# OCR MODULE (CORRECTED)
# -----------------------------

def crop_text_region(image):
    h, w = image.shape[:2]
    return image[int(h*0.4):int(h*0.9), int(w*0.1):int(w*0.9)]


def preprocess_image(image, debug=False):
    # Resize (important)
    image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Strong blur to remove foil noise
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    # Otsu threshold (best here)
    _, thresh = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Invert → text becomes white
    thresh = cv2.bitwise_not(thresh)

    if debug:
        cv2.imwrite("debug_final.png", thresh)

    return thresh


def extract_text(image):
    # Crop → reduce noise
    image = crop_text_region(image)

    # Preprocess
    processed = preprocess_image(image, debug=True)

    # OCR config (correct for structured text)
    config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(processed, config=config)

    return text


def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def extract_fields(text):
    data = {}

    # Batch
    batch = re.search(r'batch\s*no[:\-]?\s*(\w+)', text)
    if batch:
        data['batch_no'] = batch.group(1)

    # Expiry
    exp = re.search(r'(exp|expiry)[:\-]?\s*(\d{2}/\d{4})', text)
    if exp:
        data['expiry'] = exp.group(2)

    # Drug name (FIXED)
    words = re.findall(r'[a-z]{4,}', text)

    for word in words:
        match = get_close_matches(word, KNOWN_DRUGS, n=1, cutoff=0.6)
        if match:
            data['drug_name'] = match[0]
            break

    return data


# -----------------------------
# UV MODULE (unchanged)
# -----------------------------

def detect_uv_features(uv_image):
    gray = cv2.cvtColor(uv_image, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return {
        "uv_present": len(contours) > 0,
        "uv_region_count": len(contours)
    }


# -----------------------------
# DATABASE MODULE
# -----------------------------

def load_database(path):
    if not os.path.exists(path):
        print("Database not found!")
        return {}

    with open(path, "r") as f:
        return json.load(f)


# -----------------------------
# DECISION ENGINE
# -----------------------------

def verify(drug_name, extracted_data, uv_features, database):

    if not drug_name:
        return "UNKNOWN DRUG"

    drug_name = drug_name.strip()

    db_entry = database.get(drug_name)

    if not db_entry:
        return "NOT IN DATABASE"

    if db_entry.get("uv_required", False):
        if not uv_features["uv_present"]:
            return "FAKE: UV missing"

    if "batch_no" not in extracted_data:
        return "SUSPICIOUS: Batch missing"

    if "expiry" not in extracted_data:
        return "SUSPICIOUS: Expiry missing"

    return "LIKELY GENUINE"


# -----------------------------
# MAIN
# -----------------------------

def main():

    normal_img = cv2.imread(NORMAL_IMAGE_PATH)
    uv_img = cv2.imread(UV_IMAGE_PATH)

    if normal_img is None:
        print("Error loading normal image")
        return

    if uv_img is None:
        print("Error loading UV image")
        return

    # OCR
    raw_text = extract_text(normal_img)

    print("\n--- RAW OCR TEXT ---\n")
    print(raw_text)

    cleaned = clean_text(raw_text)

    print("\n--- CLEANED TEXT ---\n")
    print(cleaned)

    extracted_data = extract_fields(cleaned)

    print("\n--- EXTRACTED DATA ---\n")
    print(extracted_data)

    # UV
    uv_features = detect_uv_features(uv_img)

    print("\n--- UV FEATURES ---\n")
    print(uv_features)

    # DB
    database = load_database(DB_PATH)

    drug_name = extracted_data.get("drug_name", "")
    result = verify(drug_name, extracted_data, uv_features, database)

    print("\n--- FINAL RESULT ---\n")
    print(result)


if __name__ == "__main__":
    main()