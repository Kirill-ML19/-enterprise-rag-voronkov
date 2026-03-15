import json
import asyncio
from qdrant_client import QdrantClient

from src.indexing.embeddings import Embedder
from src.indexing.vector_store import QdrantVectorStore
from src.retrieval.retriever import Retriever
from src.retrieval.reranker import Reranker

from src.generation.llm import LLM
from src.generation.prompt import build_prompt

from src.submission.formater import build_answer
from src.submission.validator import validate_submission


async def main():

    client = QdrantClient(url="http://localhost:6333")

    embedder = Embedder()

    vector_store = QdrantVectorStore(
        client=client,
        collection_name="my_collection"
    )

    retriever = Retriever(
        embedder=embedder,
        vector_store=vector_store,
        top_k=40  
    )

    reranker = Reranker()

    llm = LLM()

    with open("data/raw/questions.json", "r", encoding="utf-8") as f:
        questions = json.load(f)

    answers = []

    for q in questions:

        question_text = q["text"]
        question_kind = q["kind"]

        print(f"\nProcessing: {question_text[:80]}")

        # -------------------------
        # Retrieval
        # -------------------------

        chunks = await retriever.retrieve(question_text)

        if not chunks:
            answers.append({
                'question_text': question_text,
                "value": "N/A",
                "references": []
            })
            continue

        # -------------------------
        # Reranking
        # -------------------------

        chunks = chunks[:25]

        chunks = await reranker.rerank(
            question_text,
            chunks
        )

        chunks = chunks[:5]

        # -------------------------
        # Prompt
        # -------------------------

        prompt = build_prompt(
            question=question_text,
            chunks=chunks,
            max_chunks=5,
            question_kind=question_kind
        )

        # -------------------------
        # LLM
        # -------------------------

        raw_answer = await llm.generate_async(
            prompt,
            timeout=60
        )

        # -------------------------
        # Formatting
        # -------------------------

        answer = build_answer(
            raw_answer=raw_answer,
            question_kind=question_kind,
            chunks=chunks
        )
        answer['question_text'] = question_text
        answers.append(answer)

        print("Raw:", raw_answer)
        print("Parsed:", answer["value"])

    submission_name = "voronkov_v4"

    submission = {
        "team_email": "test@rag-tat.com",
        "submission_name": submission_name,
        "answers": answers
    }

    try:

        validate_submission(submission)

        print("\nSubmission validated successfully")

    except Exception as e:

        print(f"\nValidation failed: {e}")
        return

    output_file = f"{submission_name}.json"

    with open(output_file, "w", encoding="utf-8") as f:

        json.dump(
            submission,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(f"\nSaved to {output_file}")


if __name__ == "__main__":
    asyncio.run(main())