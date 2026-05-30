from typing import Any, List
from phoenix.services.vector.base import BaseVectorDB
from phoenix.core.config import config

class QdrantVectorDB(BaseVectorDB):
    def __init__(self):
        self.url = config.QDRANT_URL
        self.client = None

    async def init(self):
        self.client = "MockQdrantClient"

    async def search(self, query: str) -> List[Any]:
        if not self.client:
            await self.init()
        return [f"Mock doc matching query: {query}"]

    async def insert(self, vector: Any) -> None:
        if not self.client:
            await self.init()
        pass

