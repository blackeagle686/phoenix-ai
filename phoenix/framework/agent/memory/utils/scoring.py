import math
from typing import List

def calculate_recency_score(created_at: float, decay_rate: float = 0.01) -> float:
    """Calculates a decay-based recency score."""
    import time
    time_diff = time.time() - created_at
    return math.exp(-decay_rate * time_diff)

def calculate_hybrid_score(relevance: float, importance: float, recency: float) -> float:
    """
    Standard hybrid ranking formula.
    final_score = 0.5 * relevance + 0.3 * importance + 0.2 * recency
    """
    return (0.5 * relevance) + (0.3 * importance) + (0.2 * recency)

