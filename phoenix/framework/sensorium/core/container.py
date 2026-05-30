from typing import Any, Dict, Type, TypeVar, Optional

T = TypeVar("T")

class Container:
    """
    Dependency Injection Container for Sensorium services.
    Optimized for fast service resolution.
    """
    def __init__(self):
        self._services: Dict[str, Any] = {}
        self._factories: Dict[str, Any] = {}

    def register(self, name: str, service: Any) -> None:
        """Register a singleton service instance."""
        self._services[name] = service

    def register_factory(self, name: str, factory: Any) -> None:
        """Register a factory function for a service."""
        self._factories[name] = factory

    def resolve(self, name: str) -> Any:
        """Resolve a service by name."""
        if name in self._services:
            return self._services[name]
        
        if name in self._factories:
            # Lazy initialize from factory if needed
            service = self._factories[name]()
            self._services[name] = service
            return service
            
        raise ValueError(f"Service '{name}' not found in container.")

    def has(self, name: str) -> bool:
        """Check if a service is registered."""
        return name in self._services or name in self._factories

