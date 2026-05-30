from typing import List, Optional, Tuple
import math


class SemanticCache:
    """
    In-memory semantic cache for prompt/response pairs.
    Stores query embeddings and returns cached response when similarity passes threshold.
    """

    def __init__(self, embeddings, threshold: float = 0.95):
        self.embeddings = embeddings
        self.threshold = threshold
        self._items: List[Tuple[str, str, List[float]]] = []

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    async def get_similar(self, prompt: str) -> Optional[str]:
        if not self._items:
            return None
        target = self.embeddings.embed_query(prompt)
        for _, response, emb in self._items:
            if self._cosine_similarity(target, emb) >= self.threshold:
                return response
        return None

    async def add(self, prompt: str, response: str) -> None:
        emb = self.embeddings.embed_query(prompt)
        self._items.append((prompt, response, emb))

