from typing import List
from src.types import Chunk


def build_prompt(
    question: str,
    chunks: List[Chunk],
    max_chunks: int = 5,
    question_kind: str = "text"
) -> str:

    selected_chunks = chunks[:max_chunks]

    if not selected_chunks:
        context = "No context available."
    else:
        context = "\n\n".join(
            f"[{i+1} | page {chunk.get('page_index','?')}]\n{chunk['text']}"
            for i, chunk in enumerate(selected_chunks)
        )

    kind_instruction = ""
    answer_format = ""

    if question_kind == "boolean":
        kind_instruction = "Return ONLY True or False."
        answer_format = "True | [chunk_numbers]  or  False | [chunk_numbers]"

    elif question_kind == "number":
        kind_instruction = (
            "Return ONLY the number exactly as written in the context. "
            "Do not include currency symbols, commas, or units. "
            "If the value appears in a different currency than requested, return N/A."
        )
        answer_format = "number | [chunk_numbers]  (example: 1234.56 | [2])"

    elif question_kind == "name":
        kind_instruction = (
            "Return ONLY one company or product name exactly as written in the context."
        )
        answer_format = "name | [chunk_numbers]  (example: Datalogic | [3])"

    elif question_kind == "names":
        kind_instruction = (
            "Return a comma-separated list exactly as written in the context."
        )
        answer_format = "name1, name2 | [chunk_numbers]"

    comparison_note = ""
    if any(word in question.lower() for word in ["lowest", "highest", "compare", "which of the companies"]):
        comparison_note = """
For comparison questions:
1. Identify the numeric value for each company.
2. Ignore companies where the value is missing.
3. Compare the remaining values.
4. Return ONLY the company name according to the question.
"""

    currency_note = ""
    currencies = ["USD", "EUR", "GBP", "AUD", "CAD", "JPY", "CHF", "CNY"]

    for currency in currencies:
        if f"in {currency}" in question or f"({currency})" in question:
            currency_note = f"""
IMPORTANT:
Use ONLY values explicitly reported in {currency}.
If the value appears only in another currency, return N/A.
"""
            break

    prompt = f"""
You are an information extraction system for annual financial reports.

Your task is to extract the answer ONLY from the provided context.

Rules:
1. Use ONLY the context.
2. Do NOT guess or infer.
3. If the answer is not explicitly present:
   - return N/A for numbers or names
   - return False for boolean questions
4. Return ONLY the final answer in the required format.
5. Do NOT explain your reasoning.

Answer format:
value | [chunk_numbers]

Examples:

125 | [2]

True | [3]

Datalogic | [1]

CEO, CFO | [2,4]

Where:
- value = the extracted answer
- chunk_numbers = numbers of chunks where the answer was found

Question type instruction:
{kind_instruction}

Expected format:
{answer_format}

{comparison_note}

{currency_note}

Context:
---
{context}
---

Question:
{question}

Answer:
""".strip()

    return prompt