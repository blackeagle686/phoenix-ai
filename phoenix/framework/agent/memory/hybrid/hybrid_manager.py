from typing import Optional, Dict, Any, List, Tuple
from phoenix.framework.agent.memory.short_term.stm_manager import ShortTermMemoryManager
from phoenix.framework.agent.memory.long_term.ltm_manager import LongTermMemoryManager
from phoenix.framework.chatbot.memory.semantic.semantic_search import SemanticSearch
from .hybrid_cell import HybridMemoryCell
import asyncio
import time

class HybridMemoryManager:
    """
    Unified memory manager that combines Short-Term, Long-Term, and Semantic memories
    with advanced ranking.
    """
    def __init__(self, semantic_memory_instance=None):
        # In the new architecture, we compose managers
        self.short_term = ShortTermMemoryManager()
        self.long_term = LongTermMemoryManager(semantic_memory_instance)
        
        # We'll keep these for backward compatibility if they are still needed
        # but they should eventually be migrated to task_memory or persistence
        from phoenix.framework.agent.memory.session import SessionMemory
        from phoenix.framework.agent.memory.reflection import ReflectionMemory
        self.session = SessionMemory()
        self.reflection = ReflectionMemory()
        self._search_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._search_cache_ttl = 15.0
        self._max_context_chars = 12000
        self._max_ltm_items = 6
        self._max_reflections = 3

    async def add_interaction(self, session_id: str, role: str, content: str, metadata: Optional[Dict] = None):
        await self.short_term.add(session_id, {"role": role, "content": content}, metadata=metadata)
        if self._should_store_long_term(role=role, content=content, metadata=metadata):
            await self.long_term.add(
                session_id,
                f"{role.capitalize()}: {content}",
                metadata=self._build_ltm_metadata(role=role, content=content, metadata=metadata)
            )
        if metadata:
            for key, value in metadata.items():
                self.session.set(key, value)
        self._invalidate_search_cache(session_id)

    async def get_full_context(self, session_id: str, query: str = "") -> str:
        reflections_task = asyncio.to_thread(self.reflection.get_reflections)
        session_vars_task = asyncio.to_thread(self.session.get_all)
        stm_cells_task = self.short_term.get(session_id, limit=10)
        ltm_task = self._cached_ltm_search(session_id, query)

        reflections_text, session_vars, stm_cells, ltm_results = await asyncio.gather(
            reflections_task, session_vars_task, stm_cells_task, ltm_task
        )

        ranked_ltm = self._rank_ltm_candidates(query=query, candidates=ltm_results)
        trimmed_ltm = ranked_ltm[: self._max_ltm_items]
        trimmed_reflections = self._trim_reflections(reflections_text, max_items=self._max_reflections)

        # Build context by priority with bounded size.
        context_parts: List[str] = []
        budget = self._max_context_chars

        def _append_section(title: str, body: str):
            nonlocal budget
            if not body or budget <= 0:
                return
            section = f"{title}:\n{body}"
            if len(section) <= budget:
                context_parts.append(section)
                budget -= len(section)
                return
            clipped = section[:max(0, budget)].rstrip()
            if clipped:
                context_parts.append(clipped)
            budget = 0

        ltm_text = "\n".join([self._extract_content(x) for x in trimmed_ltm if self._extract_content(x)])
        stm_text = "\n".join([f"{c.role.capitalize()}: {c.content}" for c in stm_cells]) if stm_cells else ""
        session_text = str(session_vars) if session_vars else ""

        _append_section("Relevant Past Knowledge", ltm_text)
        _append_section("Recent Conversation", stm_text)
        _append_section("Session Variables", session_text)
        _append_section("Lessons Learned", trimmed_reflections)

        return "\n\n".join(context_parts)

    def _rank_memories(self, cells: List[HybridMemoryCell]) -> List[HybridMemoryCell]:
        """
        Applies final ranking: final_score = 0.5*relevance + 0.3*importance + 0.2*recency
        """
        for cell in cells:
            cell.final_score = (
                0.5 * cell.relevance_score +
                0.3 * cell.importance_score +
                0.2 * cell.recency_score
            )
        return sorted(cells, key=lambda x: x.final_score, reverse=True)

    def _invalidate_search_cache(self, session_id: str):
        stale_keys = [k for k in self._search_cache.keys() if k[0] == session_id]
        for key in stale_keys:
            self._search_cache.pop(key, None)

    async def _cached_ltm_search(self, session_id: str, query: str):
        if not query:
            return []
        normalized_query = " ".join(query.lower().split())
        cache_key = (session_id, normalized_query)
        cached = self._search_cache.get(cache_key)
        now = time.time()
        if cached and (now - cached["ts"]) <= self._search_cache_ttl:
            return cached["results"]
        results = await self.long_term.search(session_id, query)
        self._search_cache[cache_key] = {"ts": now, "results": results}
        return results

    def _extract_content(self, item: Any) -> str:
        if hasattr(item, "content"):
            return str(item.content)
        return str(item)

    def _rank_ltm_candidates(self, query: str, candidates: List[Any]) -> List[Any]:
        if not candidates or not query:
            return candidates or []
        query_terms = set(t for t in query.lower().split() if len(t) > 2)
        ranked = []
        now = time.time()
        for candidate in candidates:
            content = self._extract_content(candidate)
            content_l = content.lower()
            overlap = len([t for t in query_terms if t in content_l])
            relevance = overlap / max(1, len(query_terms))
            importance = float(getattr(candidate, "importance_score", 0.5))
            created_at = float(getattr(candidate, "created_at", now))
            age_hours = max(0.0, (now - created_at) / 3600.0)
            recency = 1.0 / (1.0 + (age_hours / 24.0))
            final_score = (0.55 * relevance) + (0.30 * importance) + (0.15 * recency)
            ranked.append((final_score, candidate))
        ranked.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in ranked]

    def _trim_reflections(self, reflections_text: str, max_items: int = 3) -> str:
        if not reflections_text:
            return ""
        lines = [line for line in reflections_text.splitlines() if line.strip()]
        if not lines:
            return ""
        # Keep latest reflection bullets only.
        bullet_lines = [ln for ln in lines[1:] if ln.strip().startswith("-")]
        tail = bullet_lines[-max_items:] if bullet_lines else lines[-max_items:]
        return "\n".join(tail)

    def _should_store_long_term(self, role: str, content: str, metadata: Optional[Dict] = None) -> bool:
        text = (content or "").strip()
        if len(text) < 24:
            return False
        meta = metadata or {}
        if meta.get("force_long_term"):
            return True
        role_l = (role or "").lower()
        if role_l == "system":
            return True
        text_l = text.lower()
        high_value_markers = [
            "objective",
            "constraint",
            "must",
            "important",
            "final",
            "result",
            "decision",
            "preference",
            "lesson",
            "error",
            "fix",
        ]
        return any(marker in text_l for marker in high_value_markers)

    def _build_ltm_metadata(self, role: str, content: str, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        meta = dict(metadata or {})
        text = (content or "").lower()
        if "preference" in text:
            mem_type = "preference"
        elif "decision" in text or "final" in text:
            mem_type = "fact"
        elif "lesson" in text or "learned" in text:
            mem_type = "pattern"
        else:
            mem_type = meta.get("type", "knowledge")
        meta["type"] = mem_type
        meta.setdefault("importance_score", 0.65 if role.lower() == "system" else 0.55)
        return meta
