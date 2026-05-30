from phoenix.core.base import BaseService
from typing import Any, List, Optional

class BaseVectorDB(BaseService):
    async def search(self, query: str, limit: int = 5) -> List[Any]:
        raise NotImplementedError

    async def add(self, texts: List[str], metadatas: Optional[List[dict]] = None, ids: Optional[List[str]] = None) -> None:
        raise NotImplementedError

    async def delete(self, ids: List[str]) -> None:
        raise NotImplementedError

    async def clear(self) -> None:
        raise NotImplementedError
    
    async def get_by_metadata(self, where: dict) -> List[Any]:
        raise NotImplementedError

    async def insert(self, vector: Any) -> None:
        # Legacy support or specific vector insertion if needed
        raise NotImplementedError

