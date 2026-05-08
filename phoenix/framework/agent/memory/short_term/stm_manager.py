from typing import List, Optional, Dict, Any
from phoenix.framework.agent.memory.base.base_memory import BaseMemory
from .stm_cell import ShortMemoryCell
import uuid
from collections import defaultdict, deque

class ShortTermMemoryManager(BaseMemory):
    """
    Manages immediate context window memory using ShortMemoryCell.
    """
    def __init__(self, max_cells: int = 10):
        self.max_cells = max_cells
        self._session_cells: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.max_cells))

    async def add(self, session_id: str, data: Any, metadata: Optional[Dict] = None) -> None:
        """
        Expects data to be a dict with 'role' and 'content', 
        or just the content (defaulting role to 'user').
        """
        role = "user"
        content = data
        if isinstance(data, dict):
            role = data.get("role", "user")
            content = data.get("content", "")

        cell = ShortMemoryCell(
            memory_id=uuid.uuid4().hex,
            session_id=session_id,
            content=content,
            role=role,
            step=len(self._session_cells[session_id]),
            metadata=metadata or {}
        )

        self._session_cells[session_id].append(cell)

    async def get(self, session_id: str, limit: int = 10) -> List[ShortMemoryCell]:
        session_cells = self._session_cells.get(session_id)
        if not session_cells:
            return []
        return list(session_cells)[-limit:]

    async def get_context_string(self, session_id: str) -> str:
        cells = await self.get(session_id)
        return "\n".join([f"{c.role.capitalize()}: {c.content}" for c in cells])

    async def clear(self, session_id: str) -> None:
        self._session_cells.pop(session_id, None)

    async def search(self, session_id: str, query: str, limit: int = 5) -> List[ShortMemoryCell]:
        session_cells = list(self._session_cells.get(session_id, []))
        results = [c for c in session_cells if query.lower() in c.content.lower()]
        return results[-limit:]
