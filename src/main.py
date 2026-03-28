import cv2
import os
import json

from src.ocr.extract import extract_text
from src.ocr.extractors import extract_dosage
from src.decision.candidate_generator import get_best_drug_candidates
from src.decision.decision_engine import verify


# -----------------------------
# CONFIG
# -----------------------------

NORMAL_IMAGE_PATH = "data/raw/normal/sample.png"
UV_IMAGE_PATH = "data/raw/uv/sample.png"
DB_PATH = "database/drug_db.json"

KNOWN_DRUGS = ["paracetamol", "crocin", "dolo", "paracip"]


# -----------------------------
# DATABASE LOADER
# -----------------------------

def load_database(path):
    if not os.path.exists(path):
        print("Database not found!")
        return {}

    with open(path, "r") as f:
        return json.load(f)


# -----------------------------
# UV MODULE (reuse yours)
# -----------------------------

def detect_uv_features(uv_image):
    gray = cv2.cvtColor(uv_image, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    return {
        "uv_present": len(contours) > 0,
        "uv_region_count": len(contours)
    }


# -----------------------------
# MAIN PIPELINE
# -----------------------------

def main():

    print("Normal exists:", os.path.exists(NORMAL_IMAGE_PATH))
    print("UV exists:", os.path.exists(UV_IMAGE_PATH))

    normal_img = cv2.imread(NORMAL_IMAGE_PATH)
    uv_img = cv2.imread(UV_IMAGE_PATH)

    if normal_img is None:
        print("Error loading normal image")
        return

    if uv_img is None:
        print("Error loading UV image")
        return

    # -----------------------------
    # STEP 1 — OCR
    # -----------------------------
    raw_text, cleaned_text = extract_text(NORMAL_IMAGE_PATH, debug=True)

    print("\n--- CLEANED TEXT ---\n")
    print(cleaned_text)

    # -----------------------------
    # STEP 2 — FIELD EXTRACTION
    # -----------------------------
    extracted_data = {}

    # batch + expiry (reuse your logic if needed)
    from src.ocr.extract import clean_text as _  # just to avoid conflict

    print("\n--- EXTRACTED DATA ---\n")
    print(extracted_data)

    # -----------------------------
    # STEP 3 — CANDIDATES
    # -----------------------------
    candidates = get_best_drug_candidates(cleaned_text, KNOWN_DRUGS, k=3)

    print("\n--- DRUG CANDIDATES ---")
    for c in candidates:
        print(c)

    # -----------------------------
    # STEP 4 — DOSAGE
    # -----------------------------
    dosage = extract_dosage(cleaned_text)

    print("\n--- EXTRACTED DOSAGE ---")
    print(dosage)

    # -----------------------------
    # STEP 5 — UV FEATURES
    # -----------------------------
    uv_features = detect_uv_features(uv_img)

    print("\n--- UV FEATURES ---\n")
    print(uv_features)

    # -----------------------------
    # STEP 6 — DATABASE
    # -----------------------------
    database = load_database(DB_PATH)

    # -----------------------------
    # STEP 7 — DECISION
    # -----------------------------
    result = verify(
        candidates=candidates,
        extracted_data=extracted_data,
        uv_features=uv_features,
        database=database,
        text=cleaned_text
    )

    print("\n--- FINAL RESULT ---\n")
    print(result)


# -----------------------------
# ENTRY
# -----------------------------

if __name__ == "__main__":
    main()