import uuid
from qdrant_client import QdrantClient, models
from typing import List, Dict, Union
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

                sparse = item["sparse_vector"]
                if hasattr(sparse, "indices") and hasattr(sparse, "values"):
                    sparse = {
                        "indices": sparse.indices.tolist() if hasattr(sparse.indices, "tolist") else list(sparse.indices),
                        "values": sparse.values.tolist() if hasattr(sparse.values, "tolist") else list(sparse.values)
                    }

                vector_dict = {
                    "dense": item["vector"],
                    "sparse": sparse
                }

                points.append({
                    "id": point_id,
                    "vector": vector_dict,
                    "payload": payload
                })

            await self.add_points_async(points)

    async def _async_wrap(self, iterator):
        for batch in iterator:
            yield batch

    async def hybrid_search(
        self,
        dense_vector: List[float],
        sparse_vector: Union[Dict, models.SparseVector, object], 
        top_k: int = 5,
        fusion: str = "rrf"
    ) -> List[VectorPoint]:
        if hasattr(sparse_vector, "indices") and hasattr(sparse_vector, "values"):
            sparse_dict = {
                "indices": sparse_vector.indices.tolist() if hasattr(sparse_vector.indices, "tolist") else list(sparse_vector.indices),
                "values": sparse_vector.values.tolist() if hasattr(sparse_vector.values, "tolist") else list(sparse_vector.values)
            }
            sparse_vector = models.SparseVector(**sparse_dict)
        elif isinstance(sparse_vector, dict):
            sparse_vector = models.SparseVector(**sparse_vector)
        elif not isinstance(sparse_vector, models.SparseVector):
            raise TypeError("sparse_vector должен быть словарём, SparseVector или объектом с атрибутами indices/values")

        results = await asyncio.to_thread(
            self.client.query_points,
            collection_name=self.collection_name,
            prefetch=[
                models.Prefetch(query=dense_vector, using="dense", limit=top_k * 2),
                models.Prefetch(query=sparse_vector, using="sparse", limit=top_k * 2)
            ],
            query=models.FusionQuery(fusion=fusion),
            limit=top_k,
            with_payload=True,
            with_vectors=False
        )

        points = results.points if hasattr(results, 'points') else results

        seen_chunks = set()
        unique_results = []
        for r in points:
            payload = r.payload or {}
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