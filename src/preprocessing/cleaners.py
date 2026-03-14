import re

def clean_pdf_text(text: str) -> str:
    if not text or not text.strip():
        return ""

    text = re.sub(r'\n{3,}', '\n\n', text)

    text = re.sub(r'^\s*(Page|стр\.|страница)\s+\d+\s*$', '', text, flags=re.MULTILINE | re.IGNORECASE)

    text = text.strip()

    text = re.sub(r'!\[\]\(image/\d+\)', '', text) 

    return text