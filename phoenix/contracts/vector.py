
from typing import Protocol, runtime_checkable, List, Optional, Any

@runtime_checkable
class VectorStore(Protocol):
    """Protocol contract for all Vector DB adapters."""
    async def add(self, texts: List[str], metadatas: Optional[List[dict]] = None) -> None: ...
    async def search(self, query: str, limit: int = 5) -> List[Any]: ...
    async def delete(self, ids: List[str]) -> None: ...

