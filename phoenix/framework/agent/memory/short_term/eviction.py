from typing import List
from .stm_cell import ShortMemoryCell

class EvictionPolicy:
    """
    Handles memory expiration and buffer management.
    """
    @staticmethod
    def evict_expired(cells: List[ShortMemoryCell]) -> List[ShortMemoryCell]:
        from phoenix.framework.agent.memory.utils.time import has_expired
        return [c for c in cells if not has_expired(c.expires_at)]

    @staticmethod
    def truncate_to_limit(cells: List[ShortMemoryCell], limit: int) -> List[ShortMemoryCell]:
        if len(cells) <= limit:
            return cells
        return cells[-limit:]

