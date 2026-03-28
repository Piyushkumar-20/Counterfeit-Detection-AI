import re


def reconstruct_numeric_patterns(token):
    t = token.lower()

    # Do not modify already clean numbers like 500, 650mg
    if re.fullmatch(r'\d{3,4}(mg)?', t):
        return token

    # Known OCR distortion patterns → 500
    if re.search(r'(ps?00|s00|soo|5oo|s0o)', t):
        t = re.sub(r'(ps?00|s00|soo|5oo|s0o)', '500', t)

    return t


def validate_numeric_token(token):
    numbers = re.findall(r'\d+', token)

    if not numbers:
        return token

    corrected_numbers = []

    for num in numbers:
        val = int(num)

        # Valid pharma dosage set
        if val in [125, 250, 500, 650, 1000]:
            corrected_numbers.append(str(val))
        else:
            # Trim noisy values
            if str(val).startswith("500"):
                corrected_numbers.append("500")
            elif str(val).startswith("650"):
                corrected_numbers.append("650")
            elif str(val).startswith("250"):
                corrected_numbers.append("250")
            else:
                corrected_numbers.append(str(val))

    result = token
    for old, new in zip(numbers, corrected_numbers):
        result = result.replace(old, new, 1)

    return result