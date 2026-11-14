from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Any


@dataclass
class ShortMemoryCell:
    """Short-term conversational memory cell."""
    memory_id: str
    session_id: str
    content: str
    role: str = "user"
    step: int = 0
    relevance_score: float = 1.0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
