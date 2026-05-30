from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import time

@dataclass
class ShortMemoryCell:
    # ===== Identity =====
    memory_id: str
    session_id: str

    # ===== Core Content =====
    content: str
    role: str  # user | system | agent | tool

    # ===== Context =====
    step: int  # step in agent loop
    related_task_id: Optional[str] = None

    # ===== Metadata =====
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    # ===== Scoring =====
    relevance_score: float = 1.0
    importance_score: float = 0.5

    # ===== Lifecycle =====
    created_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None

    # ===== Debug =====
    source: str = "runtime"

