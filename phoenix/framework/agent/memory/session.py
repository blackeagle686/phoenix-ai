from typing import Dict, Any

class SessionMemory:
    """Maintains session state, variables, and ongoing context."""
    def __init__(self):
        self.variables: Dict[str, Any] = {}

    def set(self, key: str, value: Any):
        self.variables[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self.variables.get(key, default)

    def get_all(self) -> Dict[str, Any]:
        return self.variables.copy()

    def clear(self):
        self.variables.clear()

