from typing import Any, Dict

class Container:
    def __init__(self):
        self.services: Dict[str, Any] = {}

    def register(self, name: str, service: Any):
        self.services[name] = service

    def get(self, name: str) -> Any:
        if name not in self.services:
            raise KeyError(f"Service '{name}' not found in container.")
        return self.services[name]

container = Container()

