from typing import List, Optional, Dict, Any
from phoenix.framework.agent.memory.base.base_memory import BaseMemory
from .ltm_cell import LongMemoryCell
import uuid
import time
from collections import defaultdict

class LongTermMemoryManager(BaseMemory):
    """
    Manages long-term retrieval and storage using LongMemoryCell.
    Links with Vector DB (Chroma/Qdrant) via SemanticSearch if available.
    """
    def __init__(self, semantic_memory_instance=None):
        self.semantic = semantic_memory_instance
        self._mock_storage: Dict[str, List[LongMemoryCell]] = defaultdict(list)
        self._content_hashes: Dict[str, set] = defaultdict(set)

    async def add(self, session_id: str, data: Any, metadata: Optional[Dict] = None) -> None:
        content = str(data)
        normalized = " ".join(content.lower().split())
        if not normalized:
            return
        
        embedding = [] 
        meta = metadata or {}

        # Basic de-duplication for fallback storage path.
        content_hash = hash(normalized)
        if not self.semantic and content_hash in self._content_hashes[session_id]:
            return
        
        cell = LongMemoryCell(
            memory_id=uuid.uuid4().hex,
            content=content,
            embedding=embedding,
            memory_type=meta.get("type", "knowledge"),
            tags=meta.get("tags", []),
            source=meta.get("source", "agent"),
            source_ref=session_id
        )
        cell.importance_score = float(meta.get("importance_score", cell.importance_score))
        cell.confidence_score = float(meta.get("confidence_score", cell.confidence_score))

        if self.semantic:
            # Prevent list type inside Chroma metadata (serialize tags list to string)
            tags_raw = meta.get("tags", [])
            tags_str = ",".join(tags_raw) if isinstance(tags_raw, list) else str(tags_raw)
            
            db_meta = {
                "type": meta.get("type", "knowledge"),
                "tags": tags_str,
                "source": meta.get("source", "agent"),
                "importance_score": float(meta.get("importance_score", cell.importance_score)),
                "confidence_score": float(meta.get("confidence_score", cell.confidence_score)),
                "created_at": float(getattr(cell, "created_at", time.time()))
            }
            await self.semantic.add(session_id, content, metadata=db_meta)
        else:
            self._mock_storage[session_id].append(cell)
            self._content_hashes[session_id].add(content_hash)

    async def get(self, session_id: str, limit: int = 10) -> List[LongMemoryCell]:
        if self.semantic:
            raw_results = await self.semantic.get(session_id, limit=limit)
            cells = []
            for item in raw_results:
                if isinstance(item, dict):
                    metadata = item.get("metadata") or {}
                    tags_val = metadata.get("tags", "")
                    tags = tags_val.split(",") if tags_val else []
                    cell = LongMemoryCell(
                        memory_id=item.get("id", uuid.uuid4().hex),
                        content=item.get("content", ""),
                        embedding=[],
                        memory_type=metadata.get("type", "knowledge"),
                        tags=tags,
                        source=metadata.get("source", "agent"),
                        source_ref=metadata.get("session_id", session_id)
                    )
                    cell.importance_score = float(metadata.get("importance_score", cell.importance_score))
                    cell.confidence_score = float(metadata.get("confidence_score", cell.confidence_score))
                    cell.created_at = float(metadata.get("created_at", cell.created_at))
                    cells.append(cell)
                elif isinstance(item, LongMemoryCell):
                    cells.append(item)
            return cells
        return self._mock_storage.get(session_id, [])[-limit:]

    async def clear(self, session_id: str) -> None:
        if self.semantic:
            await self.semantic.clear(session_id)
        else:
            self._mock_storage.pop(session_id, None)
            self._content_hashes.pop(session_id, None)

    async def search(self, session_id: str, query: str, limit: int = 5) -> List[LongMemoryCell]:
        if self.semantic:
            raw_results = await self.semantic.search(session_id, query, limit=limit)
            cells = []
            for item in raw_results:
                if isinstance(item, dict):
                    metadata = item.get("metadata") or {}
                    tags_val = metadata.get("tags", "")
                    tags = tags_val.split(",") if tags_val else []
                    cell = LongMemoryCell(
                        memory_id=item.get("id", uuid.uuid4().hex),
                        content=item.get("content", ""),
                        embedding=[],
                        memory_type=metadata.get("type", "knowledge"),
                        tags=tags,
                        source=metadata.get("source", "agent"),
                        source_ref=metadata.get("session_id", session_id)
                    )
                    cell.importance_score = float(metadata.get("importance_score", cell.importance_score))
                    cell.confidence_score = float(metadata.get("confidence_score", cell.confidence_score))
                    cell.created_at = float(metadata.get("created_at", cell.created_at))
                    cells.append(cell)
                elif isinstance(item, LongMemoryCell):
                    cells.append(item)
            return cells
        
        storage = self._mock_storage.get(session_id, [])
        results = [c for c in storage if query.lower() in c.content.lower()]
        return results[:limit]
