import re

def extract_dosage(text):

    text = text.lower()

    # PRIORITY 1: dash format (more reliable)
    match = re.search(r'-(\d{2,4})', text)
    if match:
        print("[MATCH dash]", match.group(1))
        return int(match.group(1))

    # PRIORITY 2: mg format (less reliable due to OCR noise)
    match = re.search(r'(\d{2,4})\s*mg', text)
    if match:
        val = match.group(1)

        # reject weak values like 00, 000
        if int(val) < 100:
            print("[REJECTED mg - too small]", val)
        else:
            print("[MATCH mg]", val)
            return int(val)

    print("[NO VALID DOSAGE]")
    return None