"""Phoenix AI SDK package with lazy top-level exports to keep submodule imports fast."""

from importlib import import_module
from typing import Any

_MAIN_EXPORTS = frozenset(
    {
        "init_phoenix",
        "startup_phoenix",
        "init_phoenix_full",
        "get_rag_pipeline",
        "set_providers",
        "get_providers",
        "get_insight_engine",
        "get_vlm_pipeline",
        "get_llm",
        "get_memory",
    }
)

__all__ = [
    "init_phoenix",
    "startup_phoenix",
    "init_phoenix_full",
    "get_rag_pipeline",
    "set_providers",
    "get_providers",
    "get_insight_engine",
    "get_vlm_pipeline",
    "get_llm",
    "get_memory",
    "ChatBot",
    "Agent",
    "CVAgent",
]


def __getattr__(name: str) -> Any:
    if name in _MAIN_EXPORTS:
        mod = import_module(".main", __name__)
        val = getattr(mod, name)
        globals()[name] = val
        return val
    if name == "ChatBot":
        mod = import_module(".framework.chatbot", __name__)
        val = getattr(mod, "ChatBot")
        globals()[name] = val
        return val
    if name == "Agent":
        mod = import_module(".framework.agent", __name__)
        val = getattr(mod, "Agent")
        globals()[name] = val
        return val
    if name == "CVAgent":
        mod = import_module(".framework.cv_agent", __name__)
        val = getattr(mod, "CVAgent")
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


