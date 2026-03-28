import pytesseract
from .preprocess import preprocess_image

def extract_text(image_path, debug=False):
    processed = preprocess_image(image_path, debug=debug)

    config = r'--oem 3 --psm 6'

    text = pytesseract.image_to_string(processed, config=config)

    return text