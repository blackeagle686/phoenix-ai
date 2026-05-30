from abc import ABC, abstractmethod
from typing import Any, List

class BaseRetriever(ABC):
    """
    Interface for retrieval strategies.
    """
    
    @abstractmethod
    async def retrieve(self, query: str, context: Any, limit: int = 5) -> List[Any]:
        pass

