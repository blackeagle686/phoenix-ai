import requests
from phoenix.framework.agent.tools.base import BaseTool, ToolResult

class WikipediaSearchTool(BaseTool):
    name = "wikipedia_search"
    description = (
        "Searches Wikipedia for a given query and returns the summary of the best matching article. "
        "Input: 'query' (str)."
    )

    async def execute(self, query: str, **kwargs) -> ToolResult:
        try:
            # 1. Search for the best matching article title
            search_url = "https://en.wikipedia.org/w/api.php"
            search_params = {
                "action": "query",
                "list": "search",
                "srsearch": query,
                "utf8": "",
                "format": "json",
                "srlimit": 1
            }
            
            search_res = requests.get(search_url, params=search_params, timeout=10)
            search_res.raise_for_status()
            search_data = search_res.json()
            
            if not search_data.get("query", {}).get("search"):
                return ToolResult(success=False, output=None, error=f"No Wikipedia articles found for '{query}'.")
                
            best_title = search_data["query"]["search"][0]["title"]
            
            # 2. Fetch the summary for the article
            summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(best_title)}"
            summary_res = requests.get(summary_url, timeout=10)
            summary_res.raise_for_status()
            summary_data = summary_res.json()
            
            result_text = f"Title: {summary_data.get('title')}\n\n{summary_data.get('extract')}"
            
            return ToolResult(success=True, output=result_text)
            
        except Exception as e:
            return ToolResult(success=False, output=None, error=f"Wikipedia search failed: {str(e)}")
