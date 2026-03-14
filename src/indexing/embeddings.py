import logging
import asyncio
import numpy as np
from typing import List, AsyncGenerator
from sentence_transformers import SentenceTransformer
from src.types import Chunk, Embedding

logging.basicConfig(level=logging.INFO)

class Embedder:
    def __init__(self, 
                 model_name: str = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', 
                 batch_size: int = 16,
                 device: str = None
            )->None:
        
        self.model = SentenceTransformer(
            model_name_or_path=model_name,
            device=device
        )
        self.batch_size = batch_size


    def _embed_batch(self, texts: List[str]) -> np.ndarray:
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        ).astype('float32')

        return vectors.tolist()

    async def embed_chunks_stream(
            self, 
            chunks: List[Chunk]
        ) -> AsyncGenerator[Embedding, None]:

        for i in range(0, len(chunks), self.batch_size):
            batch_chunks = chunks[i: i + self.batch_size]
            texts = [chunk['text'] for chunk in batch_chunks]

            vectors = await asyncio.to_thread(
                self._embed_batch, 
                texts
            )
            results: List[Embedding] = []
            for chunk ,vector in zip(batch_chunks, vectors):
                results.append({
                    'vector': vector,
                    'text': chunk['text'],
                    'metadata': chunk['metadata']
                })
            yield results
    
    def embed_query(self, text: str) -> List[float]:

        vector = self._embed_batch([text])[0]

        return vector


    async def embed_query_async(self, text: str) -> List[float]:

        vector = await asyncio.to_thread(
            self._embed_batch,
            [text]
        )

        return vector[0]
