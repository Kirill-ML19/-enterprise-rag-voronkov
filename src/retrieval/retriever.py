import logging
from typing import List
from src.types import Chunk
from src.indexing.embeddings import Embedder
from src.indexing.vector_store import QdrantVectorStore
from fastembed import SparseTextEmbedding

logger = logging.getLogger(__name__)

class Retriever:
    def __init__(self,embedder: Embedder, vector_store: QdrantVectorStore,  top_k: int = 5):
        self.embedder = embedder
        self.sparse_model = SparseTextEmbedding('Qdrant/bm25')
        self.vector_store = vector_store
        self.top_k = top_k

    async def retrieve(self, question: str)->List[Chunk]:
        try:
           query_vector_dense = await self.embedder.embed_query_async(question)
           query_vector_sparse = list(self.sparse_model.embed(question))[0] 
           
           results = await self.vector_store.hybrid_search(
               dense_vector=query_vector_dense,
               sparse_vector=query_vector_sparse,
               top_k=self.top_k
            )

           logging.info(f'Retrieved {len(results)} chunks for query: {question[:50]}')
           return results
        
        except AttributeError as e:
            if "'QdrantClient' object has no attribute 'search'" in str(e):
                logger.error("Error: pass it on QdrantVectorStore, and not raw QdrantClient")
                logger.error("Example: vector_store = QdrantVectorStore(client, 'my_collection')")
            raise
        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            return []