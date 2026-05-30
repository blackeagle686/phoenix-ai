from phoenix.core.base import BaseService
from typing import Optional

class BaseLLM(BaseService):
    def is_available(self) -> bool:
        raise NotImplementedError

    async def generate(self, prompt: str, session_id: Optional[str] = None, max_tokens: Optional[int] = None) -> str:
        raise NotImplementedError

    async def generate_stream(self, prompt: str, session_id: Optional[str] = None, max_tokens: Optional[int] = None):
        raise NotImplementedError

class BaseVLM(BaseService):
    def is_available(self) -> bool:
        raise NotImplementedError

    async def generate_with_image(self, prompt: str, image_path: str, session_id: Optional[str] = None) -> str:
        raise NotImplementedError

    async def generate_stream(self, prompt: str, session_id: Optional[str] = None):
        raise NotImplementedError

