from typing import Any, Optional
from phoenix.core.base import BaseService

class BaseCache(BaseService):
    async def get(self, key: str) -> Optional[Any]:
        raise NotImplementedError

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        raise NotImplementedError

    async def delete(self, key: str) -> None:
        raise NotImplementedError

    async def keys(self, pattern: str = "*") -> List[str]:
        raise NotImplementedError

    async def get_all(self, pattern: str = "*") -> Dict[str, Any]:
        raise NotImplementedError
