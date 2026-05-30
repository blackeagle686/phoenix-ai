
from typing import Protocol, runtime_checkable

@runtime_checkable
class VLMProvider(Protocol):
    """Protocol contract for all VLM providers."""
    async def ask(self, prompt: str, image_path: str, **kwargs) -> str: ...
    async def describe(self, image_path: str) -> str: ...

