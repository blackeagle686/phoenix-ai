from dataclasses import dataclass, field
from typing import Any, List, Optional
import time

@dataclass
class SemanticCacheCell:
    cache_id: str

    # ===== Query =====
    query: str
    query_embedding: List[float]

    # ===== Result =====
    response: Any

    # ===== Context =====
    context_hash: str  # optional (for workspace state)

    # ===== Scoring =====
    similarity_score: float = 0.0
    hit_count: int = 0

    # ===== Lifecycle =====
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    ttl: Optional[int] = None  # seconds

