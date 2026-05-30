
import re
from .base import BaseMiddleware

class InputSanitizerMiddleware(BaseMiddleware):
    """Strips dangerous patterns and normalizes whitespace from input."""
    priority = 10

    async def process_input(self, data: str, context: dict) -> str:
        data = re.sub(r"<[^>]+>", "", data)  # Strip HTML tags
        data = " ".join(data.split())          # Normalize whitespace
        return data

    async def process_output(self, data: str, context: dict) -> str:
        return data

