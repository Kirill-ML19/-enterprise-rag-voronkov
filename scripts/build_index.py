import asyncio
from pathlib import Path
from typing import List, Dict
from src.preprocessing.chunker import chunk_md
from src.indexing.embeddings import Embedder
from src.indexing.vector_store import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, SparseVectorParams, Distance
from qdrant_client.http.exceptions import UnexpectedResponse
from fastembed import SparseTextEmbedding
from src.types import Chunk

async def process_md_files_to_qdrant(
    folder_path: str,
    qdrant_client: QdrantClient,
    collection_name: str,
    chunk_batch_size: int = 10,
    embedding_batch_size: int = 16,
):
    embedder = Embedder(batch_size=embedding_batch_size)
    sparse_model = SparseTextEmbedding("Qdrant/bm25")
    vector_dim = 384
    vector_store = QdrantVectorStore(qdrant_client, collection_name)

    try:
        qdrant_client.get_collection(collection_name=collection_name)
    except UnexpectedResponse as e:
        if e.status_code == 404:
            qdrant_client.recreate_collection(
                collection_name=collection_name,
                vectors_config={
                    "dense": VectorParams(size=vector_dim, distance=Distance.COSINE)
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams()
                }
            )

    folder = Path(folder_path)
    md_files = list(folder.glob("*.md"))

    for file_path in md_files:
        pdf_sha1 = file_path.stem
        md_text = file_path.read_text(encoding="utf-8")

        async for chunk_batch in async_wrap_chunks(md_text, pdf_sha1, chunk_batch_size):
            chunk_vectors_async = embedder.embed_chunks_stream(chunk_batch)
            dense_vectors = []
            async for batch_vectors in chunk_vectors_async:
                for emb in batch_vectors:
                    dense_vectors.append(emb['vector'])

            chunk_texts = [chunk["text"] for chunk in chunk_batch]
            sparse_vectors = list(sparse_model.embed(chunk_texts))

            batch_to_add = []
            for chunk, dense_vector, sparse_vector in zip(chunk_batch, dense_vectors, sparse_vectors):
                batch_to_add.append({
                    "text": chunk["text"],
                    "vector": dense_vector,
                    "sparse_vector": sparse_vector,
                    "metadata": chunk.get("metadata", {})
                })

            await vector_store.add_stream([batch_to_add])

        print(f"Обработан файл: {file_path.name}")

async def async_wrap_chunks(md_text: str, pdf_sha1: str, batch_size: int):
    for batch in chunk_md(md_text, pdf_sha1, batch_size=batch_size):
        yield batch

if __name__ == "__main__":
    import os
    client = QdrantClient(url="http://localhost:6333")
    asyncio.run(
        process_md_files_to_qdrant(
            folder_path="data/processed",
            qdrant_client=client,
            collection_name="my_collection"
        )
    )