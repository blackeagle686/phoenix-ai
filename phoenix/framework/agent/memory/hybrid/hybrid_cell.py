from dataclasses import dataclass
from typing import List, Optional

@dataclass
class HybridMemoryCell:
    memory_id: str

    # ===== Core =====
    content: str
    embedding: Optional[List[float]]

    # ===== Type =====
    memory_type: str  # short | long | cache

    # ===== Scoring =====
    relevance_score: float
    importance_score: float
    recency_score: float

    # ===== Final Rank =====
    final_score: float = 0.0

    # ===== Source =====
    source: str = "memory_system"

