import uuid
from qdrant_client import QdrantClient
from typing import List, Dict
import asyncio
from src.types import VectorPoint

class QdrantVectorStore:
    def __init__(self, client: QdrantClient, collection_name: str):
        self.client = client
        self.collection_name = collection_name

    async def add_points_async(self, points: List[Dict]):
        await asyncio.to_thread(
            self.client.upsert,
            collection_name=self.collection_name,
            points=points
        )

    async def add_stream(self, batches):
        if hasattr(batches, '__aiter__'):
            iterator = batches.__aiter__()
        else:
            iterator = iter(batches)

        async for batch in self._async_wrap(iterator):
            points = []

            for item in batch:
                metadata = item.get("metadata", {})

                unique_string = f"{metadata.get('pdf_sha1', '')}_{metadata.get('chunk_index', '')}"
                point_id = str(uuid.uuid5(uuid.NAMESPACE_OID, unique_string))

                payload = {
                    "text": item.get("text", ""),      
                    "chunk_id": metadata.get("chunk_id", unique_string),
                    "type": metadata.get("type", "chunk"),  
                    **{k: v for k, v in metadata.items() if k not in ["chunk_id", "type"]}
                }

                points.append({
                    "id": point_id,
                    "vector": item["vector"],
                    "payload": payload  
                })

            await self.add_points_async(points)
    
    async def _async_wrap(self, iterator):
        for batch in iterator:
            yield batch

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[VectorPoint]:
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k * 5 
        )

        seen_chunks = set()
        unique_results: List[VectorPoint] = []

        for r in results:
            payload = r.payload
            chunk_id = payload.get("chunk_id") or f"{payload.get('pdf_sha1','')}_{payload.get('chunk_index','')}"
            
            if chunk_id in seen_chunks:
                continue
            seen_chunks.add(chunk_id)

            unique_results.append({
                "vector": r.vector.tolist() if hasattr(r.vector, "tolist") else r.vector,
                "text": payload.get("text", ""), 
                "metadata": payload
            })

            if len(unique_results) >= top_k:
                break

        return unique_results