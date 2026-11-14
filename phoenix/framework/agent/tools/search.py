import asyncio
from phoenix.framework.agent.tools.base import BaseTool, ToolResult
from phoenix.framework.agent.cognition.schema import WebSearchItem, WebSearchResult


class WebSearchTool(BaseTool):
    name = "web_search"
    description = "Searches the web for information using a search engine. Input: 'query' (str), optional 'max_results' (int)."

    def _sync_search(self, query: str, max_results: int = 3):
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            return [r for r in ddgs.text(query, max_results=max_results)]

    async def execute(self, query: str, max_results: int = 3, **kwargs) -> ToolResult:
        try:
            items = []
            try:
                results = await asyncio.to_thread(self._sync_search, query, max_results)
                for r in results:
                    items.append(
                        WebSearchItem(
                            title=r.get("title", ""),
                            snippet=r.get("body", ""),
                            url=r.get("href", ""),
                        )
                    )
            except Exception as search_err:
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Web search unavailable or failed: {search_err}",
                )

            result = WebSearchResult(query=query, results=items)
            out_dict = result.model_dump() if hasattr(result, "model_dump") else result.dict()
            return ToolResult(success=True, output=out_dict)
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))
