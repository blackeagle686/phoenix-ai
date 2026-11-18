from typing import Any, Dict, Iterable, Optional
from phoenix.framework.agent.memory.budget import compact_value


class SessionMemory:
    """Lightweight bounded key-value store for session variables."""

    def __init__(self, max_vars: int = 40):
        self._vars: Dict[str, Any] = {}
        self.max_vars = max(5, max_vars)

    def set(self, key: str, value: Any):
        if key not in self._vars and len(self._vars) >= self.max_vars:
            oldest_key = next(iter(self._vars))
            self._vars.pop(oldest_key, None)
        self._vars[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self._vars.get(key, default)

    def get_all(self) -> Dict[str, Any]:
        return dict(self._vars)

    def to_context_string(
        self,
        exclude_keys: Optional[Iterable[str]] = None,
        max_chars: int = 500,
    ) -> str:
        excluded = set(exclude_keys or ())
        items = [(k, v) for k, v in self._vars.items() if k not in excluded]
        if not items:
            return ""
        formatted = ", ".join(f"{k}={compact_value(v, max_chars=100)}" for k, v in items)
        return compact_value(formatted, max_chars=max_chars)

    def clear(self):
        self._vars.clear()
