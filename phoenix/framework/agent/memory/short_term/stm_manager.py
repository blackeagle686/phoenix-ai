from typing import List, Optional, Dict, Any
from phoenix.framework.agent.memory.base.base_memory import BaseMemory
from .stm_cell import ShortMemoryCell
import uuid
from collections import defaultdict, deque

class ShortTermMemoryManager(BaseMemory):
    """
    Manages immediate context window memory using ShortMemoryCell.
    Supports Redis persistence for fault tolerance across runs.
    """
    def __init__(self, max_cells: int = 10):
        self.max_cells = max_cells
        self._session_cells: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.max_cells))
        try:
            from phoenix.core.container import container
            self._cache = container.get("cache")
        except Exception:
            self._cache = None

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

        # Try to restore existing cells from Redis first to keep correct step counting
        if not self._session_cells.get(session_id) and self._cache:
            try:
                serialized = await self._cache.get(f"stm[{session_id}]")
                if serialized:
                    cells = [ShortMemoryCell(**c) for c in serialized if isinstance(c, dict)]
                    self._session_cells[session_id].extend(cells)
            except Exception:
                pass

        cell = ShortMemoryCell(
            memory_id=uuid.uuid4().hex,
            session_id=session_id,
            content=content,
            role=role,
            step=len(self._session_cells[session_id]),
            metadata=metadata or {}
        )

        self._session_cells[session_id].append(cell)

        # Persist updated list to Redis
        if self._cache:
            try:
                cells_dict = []
                for c in self._session_cells[session_id]:
                    cells_dict.append({
                        "memory_id": c.memory_id,
                        "session_id": c.session_id,
                        "content": c.content,
                        "role": c.role,
                        "step": c.step,
                        "related_task_id": c.related_task_id,
                        "metadata": c.metadata,
                        "tags": c.tags,
                        "relevance_score": c.relevance_score,
                        "importance_score": c.importance_score,
                        "created_at": c.created_at,
                        "expires_at": c.expires_at,
                        "source": c.source
                    })
                await self._cache.set(f"stm[{session_id}]", cells_dict)
            except Exception:
                pass

    async def get(self, session_id: str, limit: int = 10) -> List[ShortMemoryCell]:
        session_cells = self._session_cells.get(session_id)
        if not session_cells and self._cache:
            try:
                serialized = await self._cache.get(f"stm[{session_id}]")
                if serialized:
                    cells = [ShortMemoryCell(**c) for c in serialized if isinstance(c, dict)]
                    self._session_cells[session_id].extend(cells)
                    session_cells = self._session_cells[session_id]
            except Exception:
                pass
        if not session_cells:
            return []
        return list(session_cells)[-limit:]

    async def get_context_string(self, session_id: str) -> str:
        cells = await self.get(session_id)
        return "\n".join([f"{c.role.capitalize()}: {c.content}" for c in cells])

    async def clear(self, session_id: str) -> None:
        self._session_cells.pop(session_id, None)
        if self._cache:
            try:
                await self._cache.delete(f"stm[{session_id}]")
            except Exception:
                pass

    async def search(self, session_id: str, query: str, limit: int = 5) -> List[ShortMemoryCell]:
        # Ensure we have restored from Redis if needed
        await self.get(session_id)
        session_cells = list(self._session_cells.get(session_id, []))
        results = [c for c in session_cells if query.lower() in c.content.lower()]
        return results[-limit:]

