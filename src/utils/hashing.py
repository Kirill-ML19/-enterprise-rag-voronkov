import hashlib

def compute_pdf_sha1(path: str)->str:
    '''
    This function hashes PDF files.

    Args:
        path (str): The path where the PDF files are located.  
    
    Returns:
        str: Returns the hash in hexadecimal format.
    '''
    sha1 = hashlib.sha1()
    with open(path, 'rb') as f:
        while chunk := f.read(8192):
            sha1.update(chunk)
    return sha1.hexdigest()