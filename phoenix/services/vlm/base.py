
from abc import ABC, abstractmethod
from typing import Optional

class BaseVLM(ABC):
    """Abstract base for all Vision-Language Model providers."""

    @abstractmethod
    async def ask(self, prompt: str, image_path: str, **kwargs) -> str:
        pass

    @abstractmethod
    async def describe(self, image_path: str) -> str:
        pass

