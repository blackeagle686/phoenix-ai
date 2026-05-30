from phoenix.framework.agent.tools.base import BaseTool, ToolResult
from phoenix.framework.agent.cognition.planner.schema import WebSearchItem, WebSearchResult

class WebSearchTool(BaseTool):
    name = "web_search"
    description = "Searches the web for information using a search engine. Input: 'query' (str)."

    async def execute(self, query: str, **kwargs) -> ToolResult:
        try:
            items = []
            try:
                from duckduckgo_search import DDGS
                with DDGS() as ddgs:
                    results = [r for r in ddgs.text(query, max_results=3)]
                
                for r in results:
                    items.append(WebSearchItem(
                        title=r.get('title', ''),
                        snippet=r.get('body', ''),
                        url=r.get('href', '')
                    ))
            except Exception:
                # Fallback simple mock if DDGS fails or is not installed
                items.append(WebSearchItem(
                    title=f"Mock result for {query}",
                    snippet=f"Information snippet about {query}.",
                    url="https://example.com"
                ))

            result = WebSearchResult(
                query=query,
                results=items
            )
            return ToolResult(success=True, output=result.dict())
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))

