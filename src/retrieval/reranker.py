import asyncio
import logging
from sentence_transformers import CrossEncoder
from src.types import Chunk
from typing import List, Tuple

class Reranker:
    def __init__(self, model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2', batch_size: int = 16):
        logging.info(f'Loading reranker model: {model_name}')
        self.model = CrossEncoder(model_name_or_path=model_name)
        self.batch_size = batch_size
    
    async def rerank(self, question:str, chunks: List[Chunk], return_scores: bool = False) -> List[Chunk] | List[Tuple[Chunk, float]]:
        if not chunks:
            return []
        
        pairs = [
            (question, chunk.get('text', ''))
            for chunk in chunks
        ]

        scores: List[float] = []

        for i in range(0,len(pairs), self.batch_size):
            batch = pairs[i:i + self.batch_size]
            try:
                batch_scores = await asyncio.to_thread(self.model.predict, batch)

                if hasattr(batch_scores, 'tolist'):
                    scores.extend(batch_scores.tolist())
                else:
                    scores.extend(batch_scores)
            except Exception as e:
                logging.error(f'Reranking error: {e}')
                return chunks if not return_scores else [(c, 0.0) for c in chunks]
        
        ranked = sorted(
            zip(chunks, scores),
            key=lambda x: x[1],
            reverse=True
        )

        logging.info(f'Reranked {len(ranked)} chunks. Top scores: {ranked[0][1]:.4f}')

        if return_scores:
            return ranked
        return [chunk for chunk, _ in ranked]