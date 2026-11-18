import inspect
import uuid
from collections import deque
from typing import Deque, Dict, List, Any, Optional
from phoenix.framework.agent.memory.budget import truncate_text
from .stm_cell import ShortMemoryCell


class ShortTermMemoryManager:
    """
    Bounded sliding-window short-term conversational memory with automatic
    zero-cost rolling compaction for evicted turns.
    """

    def __init__(
        self,
        max_cells: int = 12,
        max_message_chars: int = 1200,
        max_summary_chars: int = 900,
    ):
        self.max_cells = max(1, max_cells)
        self.max_message_chars = max(120, max_message_chars)
        self.max_summary_chars = max(200, max_summary_chars)
        self._sessions: Dict[str, Deque[ShortMemoryCell]] = {}
        self._summaries: Dict[str, str] = {}
        self._evicted_counts: Dict[str, int] = {}
        self._step_counters: Dict[str, int] = {}

    def _fold_evicted(self, session_id: str, evicted: ShortMemoryCell) -> None:
        snippet = truncate_text(evicted.content.replace("\n", " "), 140)
        line = f"{evicted.role.capitalize()}: {snippet}"
        existing = self._summaries.get(session_id, "")
        combined = f"{existing}\n{line}".strip() if existing else line
        self._summaries[session_id] = truncate_text(combined, self.max_summary_chars, mode="tail")
        self._evicted_counts[session_id] = self._evicted_counts.get(session_id, 0) + 1

    async def add(self, session_id: str, message: Dict[str, Any]) -> ShortMemoryCell:
        step = self._step_counters.get(session_id, 0) + 1
        self._step_counters[session_id] = step

        raw_content = str(message.get("content", ""))
        cell = ShortMemoryCell(
            memory_id=message.get("memory_id") or str(uuid.uuid4()),
            session_id=session_id,
            content=truncate_text(raw_content, self.max_message_chars, mode="middle"),
            role=str(message.get("role", "user")),
            step=step,
            relevance_score=float(message.get("relevance_score", 1.0)),
            pinned=bool(message.get("pinned", False)),
            metadata=dict(message.get("metadata") or {}),
        )
        bucket = self._sessions.get(session_id)
        if bucket is None:
            bucket = deque()
            self._sessions[session_id] = bucket

        if len(bucket) >= self.max_cells:
            evicted = bucket.popleft()
            self._fold_evicted(session_id, evicted)

        bucket.append(cell)
        return cell

    async def get(self, session_id: str, limit: Optional[int] = None) -> List[ShortMemoryCell]:
        bucket = self._sessions.get(session_id)
        if not bucket:
            return []
        cells = list(bucket)
        if limit is not None and limit > 0:
            return cells[-limit:]
        return cells

    def get_summary(self, session_id: str) -> str:
        return self._summaries.get(session_id, "")

    async def compress_session(self, session_id: str, llm: Any) -> Optional[str]:
        """Optional LLM-backed rolling summary compression when explicit compression is desired."""
        bucket = self._sessions.get(session_id)
        if not bucket or len(bucket) <= 4 or llm is None or not hasattr(llm, "generate"):
            return None

        # Compress older half, keep latest 4 verbatim
        cells = list(bucket)
        older, recent = cells[:-4], cells[-4:]
        older_text = "\n".join(f"{c.role.capitalize()}: {c.content}" for c in older)
        prev_summary = self._summaries.get(session_id, "")
        prompt = (
            "Summarize the conversation state into at most 4 concise bullet points (facts, decisions, user goals).\n\n"
            f"Previous Summary:\n{prev_summary}\n\nTurns to Fold:\n{older_text}"
        )
        try:
            sig = inspect.signature(llm.generate)
            if "max_tokens" in sig.parameters or any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
            ):
                new_summary = await llm.generate(prompt, max_tokens=200)
            else:
                new_summary = await llm.generate(prompt)
            if new_summary:
                self._summaries[session_id] = truncate_text(str(new_summary).strip(), self.max_summary_chars)
                self._sessions[session_id] = deque(recent)
                self._evicted_counts[session_id] = self._evicted_counts.get(session_id, 0) + len(older)
                return self._summaries[session_id]
        except Exception:
            pass
        return None

    async def get_context_string(
        self,
        session_id: str,
        max_chars: int = 2400,
        max_turns: Optional[int] = None,
    ) -> str:
        cells = await self.get(session_id, limit=max_turns)
        summary = self._summaries.get(session_id, "")
        if not cells and not summary:
            return ""

        parts: List[str] = []
        if summary:
            parts.append(f"[Earlier Summary]\n{summary}")

        if cells:
            turn_budget = max(80, min(600, max_chars // max(1, len(cells))))
            for c in cells:
                text = truncate_text(c.content, max(turn_budget, 240), mode="tail_ellipsis")
                parts.append(f"{c.role.capitalize()}: {text}")

        joined = "\n".join(parts)
        return truncate_text(joined, max_chars=max_chars, mode="tail")

    def count(self, session_id: Optional[str] = None) -> int:
        if session_id is not None:
            return len(self._sessions.get(session_id, ()))
        return sum(len(b) for b in self._sessions.values())

    def total_evicted(self, session_id: Optional[str] = None) -> int:
        if session_id is not None:
            return self._evicted_counts.get(session_id, 0)
        return sum(self._evicted_counts.values())

    def clear(self, session_id: Optional[str] = None) -> None:
        if session_id is None:
            self._sessions.clear()
            self._summaries.clear()
            self._evicted_counts.clear()
            self._step_counters.clear()
        else:
            self._sessions.pop(session_id, None)
            self._summaries.pop(session_id, None)
            self._evicted_counts.pop(session_id, None)
            self._step_counters.pop(session_id, None)
