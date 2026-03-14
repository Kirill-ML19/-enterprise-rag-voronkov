import re

def clean_pdf_text(text:str)->str:
    '''
    '''
    text = re.sub(r'-\s*\n\s*', '', text)
    text = re.sub('\n+', ' ', text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\u00A0", " ", text)
    return text.strip()