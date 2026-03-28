import re

def clean_text(text):
    # Lowercase
    text = text.lower()

    # Remove weird symbols
    text = re.sub(r'[^a-z0-9\s]', ' ', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


def extract_words(text):
    return text.split()