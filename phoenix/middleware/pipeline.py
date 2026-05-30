
from typing import List
from .base import BaseMiddleware

class MiddlewarePipeline:
    """
    Executes a sorted chain of middleware for input/output processing.
    Middleware is sorted by priority (ascending = runs first).
    """
    def __init__(self, middlewares: List[BaseMiddleware] = None):
        self._chain = sorted(middlewares or [], key=lambda m: m.priority)

    def add(self, middleware: BaseMiddleware):
        self._chain.append(middleware)
        self._chain.sort(key=lambda m: m.priority)

    async def run_input(self, data: str, context: dict = None) -> str:
        ctx = context or {}
        for m in self._chain:
            data = await m.process_input(data, ctx)
        return data

    async def run_output(self, data: str, context: dict = None) -> str:
        ctx = context or {}
        for m in reversed(self._chain):
            data = await m.process_output(data, ctx)
        return data

