import hashlib
import json
import math
import os
import re
import time
import uuid
from typing import Any, Dict, List, Optional, Set

from phoenix.framework.agent.memory.budget import truncate_text
from .ltm_cell import LongMemoryCell


_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "to", "of",
    "in", "for", "on", "with", "at", "by", "from", "and", "or", "it", "this",
    "that", "as", "if", "not", "can", "will", "do", "does", "did", "have", "has",
})


def _tokenize(text: str) -> Set[str]:
    tokens = {t for t in re.findall(r"[a-zA-Z0-9_\u0600-\u06FF]{2,}", text.lower()) if t not in _STOPWORDS}
    return tokens


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return max(0.0, dot / (na * nb))


class LongTermMemoryManager:
    """
    Lightweight, bounded, selective Long-Term Memory.
    Features deduplication, importance+recency+query scoring, and optional JSONL persistence.
    """

    def __init__(
        self,
        max_items: int = 200,
        max_item_chars: int = 360,
        storage_path: Optional[str] = None,
    ):
        self.max_items = max(10, max_items)
        self.max_item_chars = max(80, max_item_chars)
        self.storage_path = storage_path
        self._cells: Dict[str, LongMemoryCell] = {}
        self._fingerprints: Dict[str, str] = {}  # sha1 -> memory_id
        self._tokens_index: Dict[str, Set[str]] = {}
        if self.storage_path:
            self._load_from_disk()

    def _fingerprint(self, content: str) -> str:
        norm = " ".join(content.lower().split())
        return hashlib.sha1(norm.encode("utf-8")).hexdigest()

    def _load_from_disk(self) -> None:
        if not self.storage_path or not os.path.exists(self.storage_path):
            return
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    cell = LongMemoryCell.from_dict(data)
                    if cell.content:
                        fp = self._fingerprint(cell.content)
                        self._cells[cell.memory_id] = cell
                        self._fingerprints[fp] = cell.memory_id
                        self._tokens_index[cell.memory_id] = _tokenize(cell.content)
        except Exception:
            pass

    def _flush_to_disk(self) -> None:
        if not self.storage_path:
            return
        try:
            parent = os.path.dirname(os.path.abspath(self.storage_path))
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(self.storage_path, "w", encoding="utf-8") as f:
                for cell in self._cells.values():
                    f.write(json.dumps(cell.to_dict(), ensure_ascii=False) + "\n")
        except Exception:
            pass

    def add(
        self,
        content: str,
        memory_type: str = "knowledge",
        session_id: str = "",
        importance: float = 0.7,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[LongMemoryCell]:
        clean = truncate_text(str(content or "").strip(), self.max_item_chars, mode="tail_ellipsis")
        if len(clean) < 4:
            return None

        fp = self._fingerprint(clean)
        if fp in self._fingerprints:
            existing_id = self._fingerprints[fp]
            existing = self._cells.get(existing_id)
            if existing:
                existing.importance = min(1.0, max(existing.importance, importance) + 0.05)
                existing.last_accessed = time.time()
                existing.access_count += 1
                self._flush_to_disk()
                return existing

        if len(self._cells) >= self.max_items:
            self._evict_lowest_value()

        mem_id = f"ltm_{uuid.uuid4().hex[:10]}"
        cell = LongMemoryCell(
            memory_id=mem_id,
            content=clean,
            embedding=list(embedding or []),
            memory_type=memory_type,
            session_id=session_id,
            importance=max(0.0, min(1.0, float(importance))),
            metadata=dict(metadata or {}),
        )
        self._cells[mem_id] = cell
        self._fingerprints[fp] = mem_id
        self._tokens_index[mem_id] = _tokenize(clean)
        self._flush_to_disk()
        return cell

    def _evict_lowest_value(self) -> None:
        if not self._cells:
            return
        now = time.time()
        worst_id = min(
            self._cells,
            key=lambda mid: (
                self._cells[mid].importance * 0.7
                + min(0.3, self._cells[mid].access_count * 0.05)
                - min(0.4, (now - self._cells[mid].last_accessed) / 86400.0 * 0.02)
            ),
        )
        removed = self._cells.pop(worst_id, None)
        self._tokens_index.pop(worst_id, None)
        if removed:
            fp = self._fingerprint(removed.content)
            if self._fingerprints.get(fp) == worst_id:
                self._fingerprints.pop(fp, None)

    def retrieve(
        self,
        query: str = "",
        limit: int = 4,
        session_id: Optional[str] = None,
        query_embedding: Optional[List[float]] = None,
        min_score: float = 0.08,
    ) -> List[LongMemoryCell]:
        if not self._cells:
            return []

        now = time.time()
        q_tokens = _tokenize(query) if query else set()
        scored: List[tuple[float, LongMemoryCell]] = []

        for mid, cell in self._cells.items():
            # Optional session filter if caller wants strict session isolation; global facts always included
            if session_id and cell.session_id and cell.session_id != session_id and cell.memory_type == "outcome":
                continue

            age_days = max(0.0, (now - cell.created_at) / 86400.0)
            recency = 1.0 / (1.0 + 0.1 * age_days)

            if q_tokens:
                c_tokens = self._tokens_index.get(mid, set())
                overlap = len(q_tokens & c_tokens) / max(1, len(q_tokens))
                vec_sim = _cosine_similarity(query_embedding, cell.embedding) if query_embedding and cell.embedding else 0.0
                relevance = max(overlap, vec_sim)
                if cell.metadata.get("pinned"):
                    relevance = max(relevance, 0.35)
                score = 0.60 * relevance + 0.25 * cell.importance + 0.15 * recency
                if overlap == 0 and vec_sim == 0.0 and not cell.metadata.get("pinned"):
                    continue
            else:
                score = 0.65 * cell.importance + 0.35 * recency

            if score >= min_score:
                scored.append((score, cell))

        scored.sort(key=lambda item: item[0], reverse=True)
        selected = [cell for _, cell in scored[: max(1, limit)]]
        for c in selected:
            c.access_count += 1
            c.last_accessed = now
        return selected

    def get_context_string(
        self,
        query: str = "",
        limit: int = 4,
        max_chars: int = 750,
        session_id: Optional[str] = None,
    ) -> str:
        items = self.retrieve(query=query, limit=limit, session_id=session_id)
        if not items:
            return ""
        lines = [f"- [{c.memory_type}] {c.content}" for c in items]
        return truncate_text("\n".join(lines), max_chars=max_chars, mode="tail_ellipsis")

    def count(self) -> int:
        return len(self._cells)

    def clear(self) -> None:
        self._cells.clear()
        self._fingerprints.clear()
        self._tokens_index.clear()
        if self.storage_path and os.path.exists(self.storage_path):
            try:
                os.remove(self.storage_path)
            except Exception:
                pass
