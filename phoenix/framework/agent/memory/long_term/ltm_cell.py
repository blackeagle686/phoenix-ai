from dataclasses import dataclass, field
from typing import Any, Dict, List
import time


@dataclass
class LongMemoryCell:
    """Long-term selective episodic/semantic memory unit."""
    memory_id: str
    content: str
    embedding: List[float] = field(default_factory=list)
    memory_type: str = "knowledge"  # "fact" | "lesson" | "preference" | "knowledge" | "outcome"
    session_id: str = ""
    importance: float = 0.7
    access_count: int = 0
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "content": self.content,
            "embedding": self.embedding,
            "memory_type": self.memory_type,
            "session_id": self.session_id,
            "importance": self.importance,
            "access_count": self.access_count,
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LongMemoryCell":
        return cls(
            memory_id=str(data.get("memory_id", "")),
            content=str(data.get("content", "")),
            embedding=list(data.get("embedding") or []),
            memory_type=str(data.get("memory_type", "knowledge")),
            session_id=str(data.get("session_id", "")),
            importance=float(data.get("importance", 0.7)),
            access_count=int(data.get("access_count", 0)),
            created_at=float(data.get("created_at", time.time())),
            last_accessed=float(data.get("last_accessed", time.time())),
            metadata=dict(data.get("metadata") or {}),
        )
