"""Resilient web search tool supporting both `ddgs` and `duckduckgo_search` with timeout protection."""

from __future__ import annotations

import asyncio
from phoenix.framework.agent.cognition.schema import WebSearchItem, WebSearchResult
from phoenix.framework.agent.tools.base import BaseTool, ToolResult


class WebSearchTool(BaseTool):
    name = "web_search"
    read_only = True
    description = "Searches the web for information using DuckDuckGo. Inputs: 'query' (str), optional 'max_results' (int=3)."

    def __init__(self, timeout: float = 15.0):
        super().__init__(read_only=True, timeout=timeout)

    def _sync_search(self, query: str, max_results: int = 3):
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=max_results))

    async def execute(self, query: str, max_results: int = 3, **kwargs) -> ToolResult:
        try:
            clean_query = str(query or "").strip()
            if not clean_query:
                return ToolResult(success=False, output=None, error="Empty search 'query' provided.")

            cap = max(1, min(int(max_results or 3), 20))
            timeout_sec = float(self.timeout or 15.0)
            try:
                results = await asyncio.wait_for(
                    asyncio.to_thread(self._sync_search, clean_query, cap),
                    timeout=timeout_sec,
                )
            except asyncio.TimeoutError:
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Web search timed out after {timeout_sec}s.",
                )
            except Exception as search_err:
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Web search unavailable or failed: {search_err}",
                )

            items = [
                WebSearchItem(
                    title=r.get("title", ""),
                    snippet=r.get("body", "")[:400],
                    url=r.get("href", ""),
                )
                for r in (results or [])
            ]
            result = WebSearchResult(query=clean_query, results=items)
            out_dict = result.model_dump() if hasattr(result, "model_dump") else result.dict()
            return ToolResult(success=True, output=out_dict)
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))
