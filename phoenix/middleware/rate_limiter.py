
import time
from collections import defaultdict
from .base import BaseMiddleware

class RateLimiterMiddleware(BaseMiddleware):
    """Simple in-memory per-session rate limiter."""
    priority = 5

    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window = window_seconds
        self._counts = defaultdict(list)

    async def process_input(self, data: str, context: dict) -> str:
        session_id = context.get("session_id", "global")
        now = time.time()
        self._counts[session_id] = [t for t in self._counts[session_id] if now - t < self.window]
        if len(self._counts[session_id]) >= self.max_requests:
            raise RuntimeError(f"Rate limit exceeded for session '{session_id}'.")
        self._counts[session_id].append(now)
        return data

    async def process_output(self, data: str, context: dict) -> str:
        return data

