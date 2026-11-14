from typing import Any, Dict


class SessionMemory:
    """Lightweight key-value store for session variables."""

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
