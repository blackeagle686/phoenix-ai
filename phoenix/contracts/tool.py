
from typing import Protocol, runtime_checkable, Any, Dict

@runtime_checkable
class ToolContract(Protocol):
    """Protocol contract for agent tools."""
    name: str
    description: str
    async def run(self, **kwargs: Any) -> Any: ...
    def schema(self) -> Dict: ...

