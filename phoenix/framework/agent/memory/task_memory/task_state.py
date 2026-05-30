from dataclasses import dataclass, field
from typing import Any, List, Optional
import time

@dataclass
class TaskMemoryCell:
    task_id: str

    # ===== Core =====
    description: str

    # ===== Execution =====
    status: str  # pending | in_progress | done | failed
    priority: str  # high | medium | low

    # ===== Dependencies =====
    dependencies: List[str] = field(default_factory=list)

    # ===== Tools =====
    tools_required: List[str] = field(default_factory=list)

    # ===== Results =====
    result: Optional[Any] = None
    error: Optional[str] = None

    # ===== Tracking =====
    attempts: int = 0

    # ===== Lifecycle =====
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

