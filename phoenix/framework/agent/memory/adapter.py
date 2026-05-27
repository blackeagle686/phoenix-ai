from typing import Any, Dict, Optional


class _SessionProxy:
    def __init__(self):
        self._vars: Dict[str, Any] = {}

    def set(self, key: str, value: Any):
        self._vars[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self._vars.get(key, default)

    def get_all(self) -> Dict[str, Any]:
        return dict(self._vars)

    def clear(self):
        self._vars.clear()


class _ReflectionProxy:
    def __init__(self):
        self._items = []

    def add_reflection(self, reflection: str):
        self._items.append(reflection)

    def get_reflections(self) -> str:
        if not self._items:
            return ""
        return "Lessons Learned:\n" + "\n".join([f"- {x}" for x in self._items])

    async def consolidate(self, llm):
        return None

    def clear(self):
        self._items.clear()


class InteractiveMemoryAdapter:
    """
    Adapts user-provided memory implementations to the Agent memory contract.

    Expected Agent contract:
    - async add_interaction(session_id, role, content, metadata=None)
    - async get_full_context(session_id, query="")
    - async consolidate_reflections(llm)
    - session.{set,get,get_all,clear}
    - reflection.{add_reflection,get_reflections,consolidate,clear}

    Supported custom memory entry points (any of):
    - async add_interaction(...)
    - async add_message(...)
    - async store(...)
    and
    - async get_full_context(...)
    - async get_context(...)
    """

    def __init__(self, custom_memory: Any):
        self._custom = custom_memory
        self.session = getattr(custom_memory, "session", _SessionProxy())
        self.reflection = getattr(custom_memory, "reflection", _ReflectionProxy())

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
            return await self._custom.get_context(session_id=session_id, query=query)
        return ""

    async def consolidate_reflections(self, llm, session_id: Optional[str] = None):
        if hasattr(self._custom, "consolidate_reflections"):
            import inspect
            sig = inspect.signature(self._custom.consolidate_reflections)
            if "session_id" in sig.parameters:
                return await self._custom.consolidate_reflections(llm, session_id=session_id)
            return await self._custom.consolidate_reflections(llm)
        if hasattr(self.reflection, "consolidate"):
            return await self.reflection.consolidate(llm)
        return None

