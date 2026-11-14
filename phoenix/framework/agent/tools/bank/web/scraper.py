import asyncio
import requests
from phoenix.framework.agent.tools.base import BaseTool, ToolResult

class WebScraperTool(BaseTool):
    name = "web_scraper"
    description = (
        "Fetches the text content of a web page given its URL. "
        "Input: 'url' (str)."
    )

    async def execute(self, url: str, **kwargs) -> ToolResult:
        def _sync_scrape() -> ToolResult:
            try:
                try:
                    from bs4 import BeautifulSoup
                except ImportError:
                    return ToolResult(
                        success=False,
                        output=None,
                        error="BeautifulSoup is required for this tool. Please run: pip install beautifulsoup4"
                    )

                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                }

                response = requests.get(url, headers=headers, timeout=15)
                response.raise_for_status()

                soup = BeautifulSoup(response.content, "html.parser")
                for script in soup(["script", "style", "nav", "footer", "header"]):
                    script.decompose()

                text = soup.get_text(separator=' ', strip=True)
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = '\n'.join(chunk for chunk in chunks if chunk)

                return ToolResult(success=True, output=text[:15000])
            except Exception as e:
                return ToolResult(success=False, output=None, error=f"Failed to scrape {url}: {str(e)}")

        return await asyncio.to_thread(_sync_scrape)

