import inspect
from typing import Any, Optional
from phoenix.framework.agent.memory.session import SessionMemory
from phoenix.framework.agent.memory.reflection import ReflectionMemory


class InteractiveMemoryAdapter:
    """
    Adapts user-provided memory implementations to the Agent memory contract.

    Expected Agent contract:
    - async add_interaction(session_id, role, content, metadata=None)
    - async get_full_context(session_id, query="")
    - async consolidate_reflections(llm)
    - session.{set,get,get_all,clear}
    - reflection.{add_reflection,get_reflections,consolidate,clear}
    """

    def __init__(self, custom_memory: Any):
        self._custom = custom_memory
        self.session = getattr(custom_memory, "session", SessionMemory())
        self.reflection = getattr(custom_memory, "reflection", ReflectionMemory())

    async def add_interaction(self, session_id: str, role: str, content: str, metadata: Optional[dict] = None):
        if hasattr(self._custom, "add_interaction"):
            return await self._custom.add_interaction(session_id, role, content, metadata=metadata)
        if hasattr(self._custom, "add_message"):
            return await self._custom.add_message(session_id=session_id, role=role, content=content, metadata=metadata)
        if hasattr(self._custom, "store"):
            return await self._custom.store(session_id=session_id, role=role, content=content, metadata=metadata)
        raise AttributeError("Custom memory must provide add_interaction/add_message/store")

    async def get_full_context(self, session_id: str, query: str = "") -> str:
        if hasattr(self._custom, "get_full_context"):
            return await self._custom.get_full_context(session_id=session_id, query=query)
        if hasattr(self._custom, "get_context"):
            sig = inspect.signature(self._custom.get_context)
            if "query" in sig.parameters or any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
            ):
                return await self._custom.get_context(session_id=session_id, query=query)
            return await self._custom.get_context(session_id)
        return ""

    async def consolidate_reflections(self, llm, session_id: Optional[str] = None):
        if hasattr(self._custom, "consolidate_reflections"):
            sig = inspect.signature(self._custom.consolidate_reflections)
            if "session_id" in sig.parameters:
                return await self._custom.consolidate_reflections(llm, session_id=session_id)
            return await self._custom.consolidate_reflections(llm)
        if hasattr(self.reflection, "consolidate"):
            return await self.reflection.consolidate(llm)
        return None
