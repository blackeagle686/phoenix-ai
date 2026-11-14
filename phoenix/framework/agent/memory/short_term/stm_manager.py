import uuid
from typing import Dict, List, Any
from .stm_cell import ShortMemoryCell


class ShortTermMemoryManager:
    """Bounded FIFO/sliding-window short-term conversational memory manager."""

    def __init__(self, max_cells: int = 20):
        self.max_cells = max(1, max_cells)
        self._sessions: Dict[str, List[ShortMemoryCell]] = {}
        self._step_counters: Dict[str, int] = {}

    async def add(self, session_id: str, message: Dict[str, Any]) -> ShortMemoryCell:
        step = self._step_counters.get(session_id, 0) + 1
        self._step_counters[session_id] = step

        cell = ShortMemoryCell(
            memory_id=message.get("memory_id") or str(uuid.uuid4()),
            session_id=session_id,
            content=str(message.get("content", "")),
            role=str(message.get("role", "user")),
            step=step,
            relevance_score=float(message.get("relevance_score", 1.0)),
            metadata=dict(message.get("metadata") or {}),
        )
        bucket = self._sessions.setdefault(session_id, [])
        bucket.append(cell)
        if len(bucket) > self.max_cells:
            del bucket[: len(bucket) - self.max_cells]
        return cell

    async def get(self, session_id: str) -> List[ShortMemoryCell]:
        return list(self._sessions.get(session_id, []))

    async def get_context_string(self, session_id: str) -> str:
        cells = await self.get(session_id)
        return "\n".join(f"{c.role.capitalize()}: {c.content}" for c in cells)
