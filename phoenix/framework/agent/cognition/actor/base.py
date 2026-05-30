from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseActor(ABC):
    def __init__(self, tool_manager):
        self.tool_manager = tool_manager

    @abstractmethod
    async def execute(self, plan: Dict[str, Any]) -> str:
        pass

