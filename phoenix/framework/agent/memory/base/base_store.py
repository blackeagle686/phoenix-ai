from abc import ABC, abstractmethod
from typing import Any, List, Optional, Dict

class BaseStore(ABC):
    """
    Interface for storage backends.
    """
    
    @abstractmethod
    async def save(self, key: str, data: Any) -> None:
        pass

    @abstractmethod
    async def load(self, key: str) -> Any:
        pass

    @abstractmethod
    async def delete(self, key: str) -> None:
        pass

    @abstractmethod
    async def list_keys(self) -> List[str]:
        pass

