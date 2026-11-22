"""Phoenix framework exports with lazy module loading so agent imports stay fast and lightweight."""

from importlib import import_module
from typing import Any

_EXPORTS = {
    "ChatBot": (".chatbot.core", "ChatBot"),
    "Agent": (".agent.core.agent", "Agent"),
    "CVAgent": (".cv_agent.agent", "CVAgent"),
    "MicroCVAgent": (".cv_agent.agent", "MicroCVAgent"),
    "CVAgentConfig": (".cv_agent.agent", "CVAgentConfig"),
    "CVEvent": (".cv_agent.events", "CVEvent"),
    "Detection": (".cv_agent.events", "Detection"),
    "MicroMoELLM": (".cv_agent.moe_model", "MicroMoELLM"),
    "MultiAgentManager": (".multi_agent.manager", "MultiAgentManager"),
    "MultiAgentConfig": (".multi_agent.config", "MultiAgentConfig"),
    "AgentConfig": (".multi_agent.config", "AgentConfig"),
    "AgentMessage": (".multi_agent.protocol", "AgentMessage"),
    "MessageType": (".multi_agent.protocol", "MessageType"),
    "Priority": (".multi_agent.protocol", "Priority"),
    "MessageBus": (".multi_agent.message_bus", "MessageBus"),
    "SharedStateStore": (".multi_agent.state_store", "SharedStateStore"),
    "RAG": (".rag", "RAG"),
    "CAG": (".rag", "CAG"),
    "AgenticRAG": (".rag", "AgenticRAG"),
    "AdaptiveRAG": (".rag", "AdaptiveRAG"),
    "MultiModalRAG": (".rag", "MultiModalRAG"),
    "BaseRAG": (".rag", "BaseRAG"),
    "RAGConfig": (".rag.config", "RAGConfig"),
    "CAGConfig": (".rag.config", "CAGConfig"),
    "AgenticRAGConfig": (".rag.config", "AgenticRAGConfig"),
    "MultiModalRAGConfig": (".rag.config", "MultiModalRAGConfig"),
}

__all__ = list(_EXPORTS.keys())


def __getattr__(name: str) -> Any:
    spec = _EXPORTS.get(name)
    if spec is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod_path, attr = spec
    mod = import_module(mod_path, __name__)
    val = getattr(mod, attr)
    globals()[name] = val
    return val
