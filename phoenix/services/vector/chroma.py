try:
    import chromadb
except ImportError:
    chromadb = None
from typing import Any, List, Optional
from phoenix.services.vector.base import BaseVectorDB
from phoenix.core.config import config
from phoenix.services.observability.logger import get_logger

logger = get_logger("Phoenix AI.Vector.Chroma")

class ChromaVectorDB(BaseVectorDB):
    def __init__(self, collection_name: str = "PhoenixAI_collection", embedding_service=None):
        self.persist_directory = config.CHROMA_PERSIST_DIR
        self.collection_name = collection_name
        self.embedding_service = embedding_service
        self.client = None
        self.collection = None

    async def init(self):
        if chromadb is None:
            logger.warning("chromadb is not installed. Vector DB disabled. Install using pip install phx-ashborn[vector]")
            self._failed = True
            return
        self._failed = False
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        
        # Link our embedding service to Chroma's interface if provided
        embedding_function = None
        if self.embedding_service:
            from chromadb import EmbeddingFunction, Documents, Embeddings
            class SDKEmbeddingWrapper(EmbeddingFunction):
                def __init__(self, service):
                    self.service = service
                def __call__(self, input: Documents) -> Embeddings:
                    return self.service.embed_documents(input)
            
            embedding_function = SDKEmbeddingWrapper(self.embedding_service)

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=embedding_function
        )

    async def search(self, query: str, limit: int = 5, where: Optional[dict] = None) -> List[Any]:
        if not self.collection:
            await self.init()
        
        results = self.collection.query(
            query_texts=[query],
            n_results=limit,
            where=where
        )
        # Flatten results to a list of dicts or documents
        docs = []
        for i in range(len(results['documents'][0])):
            docs.append({
                "content": results['documents'][0][i],
                "metadata": results['metadatas'][0][i] if results['metadatas'] else {},
                "id": results['ids'][0][i],
                "distance": results['distances'][0][i] if 'distances' in results else None
            })
        return docs

    async def add(self, texts: List[str], metadatas: Optional[List[dict]] = None, ids: Optional[List[str]] = None) -> None:
        if not self.collection:
            await self.init()
        
        if not ids:
            import uuid
            ids = [str(uuid.uuid4()) for _ in texts]
        
        # Batching for progress reporting and large inserts
        batch_size = 500
        total = len(texts)
        
        import asyncio
        # Limit concurrency to 4 to balance PyTorch VRAM / CPU limits and SQLite lock contention
        semaphore = asyncio.Semaphore(4)
        
        async def add_batch(start: int, end: int):
            async with semaphore:
                def _do_add():
                    self.collection.add(
                        documents=texts[start:end],
                        metadatas=metadatas[start:end] if metadatas else None,
                        ids=ids[start:end]
                    )
                await asyncio.to_thread(_do_add)
                logger.info(f"Progress: {end}/{total} units indexed...")

        tasks = []
        for i in range(0, total, batch_size):
            end = min(i + batch_size, total)
            tasks.append(add_batch(i, end))
            
        if tasks:
            await asyncio.gather(*tasks)


    async def delete(self, ids: List[str]) -> None:
        if not self.collection:
            await self.init()
        self.collection.delete(ids=ids)

    async def clear(self) -> None:
        if not self.collection:
            await self.init()
        ids = self.collection.get()['ids']
        if ids:
            self.collection.delete(ids=ids)

    async def get_all(self) -> List[Any]:
        if not self.collection:
            await self.init()
        return self.collection.get()

    async def get_by_metadata(self, where: dict) -> List[Any]:
        if not self.collection:
            await self.init()
        results = self.collection.get(where=where)
        docs = []
        for i in range(len(results['documents'])):
            docs.append({
                "content": results['documents'][i],
                "metadata": results['metadatas'][i] if results['metadatas'] else {},
                "id": results['ids'][i]
            })
        return docs

    async def insert(self, vector: Any) -> None:
        # Placeholder for raw vector insertion if needed
        pass
