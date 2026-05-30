from phoenix.core.base import BaseService
from typing import Any

class BaseQueue(BaseService):
    async def enqueue(self, task_name: str, *args, **kwargs) -> Any:
        raise NotImplementedError

