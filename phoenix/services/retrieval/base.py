from phoenix.core.base import BaseService
from typing import Optional

class BaseInsightService(BaseService):
    async def query(self, question: str, context: Optional[dict] = None):
        """Execute a query through the insight ecosystem."""
        raise NotImplementedError

