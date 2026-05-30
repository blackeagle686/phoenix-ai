from phoenix.core.base import BaseService
from typing import Any

class BaseDB(BaseService):
    async def execute(self, query: str, *args, **kwargs) -> Any:
        raise NotImplementedError

    async def fetch_one(self, query: str, *args, **kwargs) -> Any:
        raise NotImplementedError

    async def fetch_all(self, query: str, *args, **kwargs) -> Any:
        raise NotImplementedError

