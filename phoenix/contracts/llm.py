
from typing import Protocol, runtime_checkable, AsyncIterator

@runtime_checkable
class LLMProvider(Protocol):
    """Protocol contract for all LLM providers."""
    async def generate(self, prompt: str, **kwargs) -> str: ...
    async def stream(self, prompt: str, **kwargs) -> AsyncIterator[str]: ...

