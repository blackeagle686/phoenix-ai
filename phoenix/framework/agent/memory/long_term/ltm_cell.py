from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import time

@dataclass
class LongMemoryCell:
    # ===== Identity =====
    memory_id: str

    # ===== Core Content =====
    content: str
    embedding: List[float]

    # ===== Classification =====
    memory_type: str  # knowledge | preference | fact | pattern
    tags: List[str] = field(default_factory=list)

    # ===== Source =====
    source: str = "agent"  # user | file | web | system
    source_ref: Optional[str] = None

    # ===== Scoring =====
    importance_score: float = 0.5
    relevance_score: float = 0.0
    confidence_score: float = 0.8

    # ===== Usage Tracking =====
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)

    # ===== Lifecycle =====
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    # ===== Relationships =====
    related_memories: List[str] = field(default_factory=list)

