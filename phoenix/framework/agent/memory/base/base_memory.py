from abc import ABC, abstractmethod
from typing import Any, List, Optional, Dict

class BaseMemory(ABC):
    """
    Base class for all memory components in Phoenix AI SDK.
    """
    
    @abstractmethod
    async def add(self, session_id: str, data: Any, metadata: Optional[Dict] = None) -> None:
        """Add data to memory."""
        pass

    @abstractmethod
    async def get(self, session_id: str, limit: int = 10) -> List[Any]:
        """Retrieve recent data from memory."""
        pass

    @abstractmethod
    async def clear(self, session_id: str) -> None:
        """Clear memory."""
        pass

    @abstractmethod
    async def search(self, session_id: str, query: str, limit: int = 5) -> List[Any]:
        """Search across memory."""
        pass

