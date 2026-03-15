import re
from typing import List
from src.types import Chunk, Answer, References


def parse_chunk_indices(raw_answer: str):

    match = re.search(r'\[(.*?)\]', raw_answer)

    if not match:
        return []

    inside = match.group(1)

    numbers = re.findall(r'\d+', inside)

    return [int(n) for n in numbers]


def extract_references_from_indices(
    chunks: List[Chunk],
    indices: List[int],
    max_refs: int = 2
) -> List[References]:

    refs: List[References] = []
    seen = set()

    for idx in indices:

        if idx <= 0 or idx > len(chunks):
            continue

        chunk = chunks[idx - 1]

        pdf_sha1 = chunk.get('metadata', {}).get('pdf_sha1')
        page_index = chunk.get('metadata', {}).get('page_index')

        if pdf_sha1 is None or page_index is None:
            continue

        key = (pdf_sha1, page_index)

        if key in seen:
            continue

        refs.append({
            "pdf_sha1": pdf_sha1,
            "page_index": int(page_index)
        })

        seen.add(key)

        if len(refs) >= max_refs:
            break

    return refs


def split_answer(raw_answer: str):

    if "|" not in raw_answer:
        return raw_answer.strip(), []

    value_part, ref_part = raw_answer.split("|", 1)

    indices = parse_chunk_indices(ref_part)

    return value_part.strip(), indices


def _parse_number(text: str):

    cleaned = re.sub(r'[^\d.,\-]', '', text.replace(' ', ''))

    if ',' in cleaned and '.' in cleaned:
        cleaned = cleaned.replace(',', '')
    elif ',' in cleaned and not '.' in cleaned:
        cleaned = cleaned.replace('.', '').replace(',', '.')
    else:
        cleaned = cleaned.replace(',', '')

    match = re.search(r'-?\d+\.?\d*', cleaned)

    if not match:
        return 'N/A'

    try:
        return float(match.group())
    except ValueError:
        return 'N/A'


def format_value(raw_answer: str, kind: str):

    if not raw_answer:
        return 'N/A'

    raw_lower = raw_answer.strip().lower()

    if raw_lower in {'n/a', 'not found', 'not available', 'no data'}:
        return 'N/A'

    if kind == 'number':
        return _parse_number(raw_answer)

    if kind == 'boolean':

        if raw_lower in {'true', 'yes', '1', 'True'}:
            return 'True'

        if raw_lower in {'false', 'no', '0', 'False'}:
            return 'False'

        return 'N/A'

    if kind == 'name':
        cleaned = raw_answer.strip().strip('"').strip("'")
        return cleaned if cleaned else 'N/A'

    if kind == 'names':
        parts = raw_answer.replace(' and ', ',').split(',')
        names = [n.strip().strip('"').strip("'") for n in parts if n.strip()]
        return names if names else 'N/A'

    return raw_answer.strip()


def build_answer(raw_answer: str, question_kind: str, chunks: List[Chunk]) -> Answer:

    value_part, indices = split_answer(raw_answer)

    value = format_value(value_part, question_kind)

    references = extract_references_from_indices(chunks, indices)

    return {
        "value": value,
        "references": references
    }