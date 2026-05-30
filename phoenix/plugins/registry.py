
from typing import Dict, Type, Any

class PluginRegistry:
    """
    Central registry for discovering and loading Phoenix plugins.
    Supports LLM providers, Vector DB adapters, Memory backends, and Tools.
    """
    _plugins: Dict[str, Dict[str, Any]] = {
        "llm": {},
        "vlm": {},
        "vector": {},
        "memory": {},
        "tool": {},
    }

    @classmethod
    def register(cls, category: str, name: str, plugin_class: Type):
        """Register a plugin class under a category with a given name."""
        if category not in cls._plugins:
            cls._plugins[category] = {}
        cls._plugins[category][name] = plugin_class

    @classmethod
    def get(cls, category: str, name: str) -> Type:
        """Retrieve a registered plugin class."""
        try:
            return cls._plugins[category][name]
        except KeyError:
            raise ValueError(f"Plugin '{name}' not found in category '{category}'.")

    @classmethod
    def list_plugins(cls, category: str = None) -> dict:
        """List all registered plugins or those in a specific category."""
        if category:
            return list(cls._plugins.get(category, {}).keys())
        return {k: list(v.keys()) for k, v in cls._plugins.items()}

registry = PluginRegistry()

