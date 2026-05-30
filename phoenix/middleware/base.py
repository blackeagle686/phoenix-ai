
from abc import ABC, abstractmethod
from typing import Any, Optional

class BaseMiddleware(ABC):
    """Abstract base class for all Phoenix middleware components."""
    priority: int = 100  # Lower runs first

    @abstractmethod
    async def process_input(self, data: str, context: dict) -> str:
        """Transform or validate incoming request data."""
        return data

    @abstractmethod
    async def process_output(self, data: str, context: dict) -> str:
        """Transform or filter outgoing response data."""
        return data

