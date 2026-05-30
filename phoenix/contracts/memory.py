
from typing import Protocol, runtime_checkable, Optional, Dict

@runtime_checkable
class MemoryBackend(Protocol):
    """Protocol contract for memory backends."""
    async def add_interaction(self, session_id: str, role: str, content: str, metadata: Optional[Dict] = None) -> None: ...
    async def get_context(self, session_id: str) -> str: ...
    async def clear_session(self, session_id: str) -> None: ...

