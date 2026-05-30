import requests
import xml.etree.ElementTree as ET
from phoenix.framework.agent.tools.base import BaseTool, ToolResult

class ArxivSearchTool(BaseTool):
    name = "arxiv_search"
    description = (
        "Searches the arXiv academic database for research papers and returns their titles, authors, and abstracts. "
        "Input: 'query' (str, the topic or keywords to search for), 'max_results' (int, optional, default 3)."
    )

    async def execute(self, query: str, max_results: int = 3, **kwargs) -> ToolResult:
        try:
            # ArXiv API uses a specific query format, we use standard 'all:' search
            formatted_query = f"all:{query.replace(' ', '+')}"
            
            url = "http://export.arxiv.org/api/query"
            params = {
                "search_query": formatted_query,
                "start": 0,
                "max_results": max_results,
                "sortBy": "relevance",
                "sortOrder": "descending"
            }
            
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            
            # The API returns XML (Atom feed format)
            root = ET.fromstring(response.content)
            namespace = {'atom': 'http://www.w3.org/2005/Atom'}
            
            entries = root.findall('atom:entry', namespace)
            
            if not entries:
                return ToolResult(success=True, output=f"No arXiv papers found for query: '{query}'.")
                
            results = []
            for entry in entries:
                title = entry.find('atom:title', namespace).text.replace('\n', ' ').strip()
                summary = entry.find('atom:summary', namespace).text.replace('\n', ' ').strip()
                published = entry.find('atom:published', namespace).text
                
                authors = []
                for author in entry.findall('atom:author', namespace):
                    name = author.find('atom:name', namespace).text
                    authors.append(name)
                    
                paper_id = entry.find('atom:id', namespace).text
                
                results.append(
                    f"Title: {title}\n"
                    f"Authors: {', '.join(authors)}\n"
                    f"Published: {published[:10]}\n"
                    f"Link: {paper_id}\n"
                    f"Abstract: {summary}\n"
                )
                
            return ToolResult(success=True, output="\n---\n".join(results))
            
        except Exception as e:
            return ToolResult(success=False, output=None, error=f"ArXiv search failed: {str(e)}")

