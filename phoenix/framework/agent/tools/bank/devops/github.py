import os
import base64
import requests
from typing import Optional
from phoenix.framework.agent.tools.base import BaseTool, ToolResult

class GitHubTool(BaseTool):
    name = "github"
    description = (
        "Interacts with GitHub via its REST API. "
        "Inputs: 'action' (str, one of: 'search_repos', 'get_file', 'get_issue', 'list_commits'), "
        "'repo' (str, e.g., 'owner/repo'), "
        "'path' (str, file path for get_file), "
        "'issue_number' (int, for get_issue), "
        "'query' (str, for search_repos)."
    )

    async def execute(
        self, 
        action: str, 
        repo: Optional[str] = None, 
        path: Optional[str] = None, 
        issue_number: Optional[int] = None, 
        query: Optional[str] = None,
        github_token: Optional[str] = None,
        **kwargs
    ) -> ToolResult:
        try:
            # Look for token in environment if not provided
            token = github_token or os.environ.get("GITHUB_TOKEN")
            
            headers = {
                "Accept": "application/vnd.github.v3+json",
                "User-Agent": "PhoenixAI/1.0"
            }
            if token:
                headers["Authorization"] = f"token {token}"
                
            base_url = "https://api.github.com"
            
            if action == "search_repos":
                if not query:
                    return ToolResult(success=False, output=None, error="Missing 'query' for search_repos.")
                
                url = f"{base_url}/search/repositories"
                res = requests.get(url, headers=headers, params={"q": query, "per_page": 5}, timeout=10)
                res.raise_for_status()
                
                items = res.json().get("items", [])
                output = [f"Repo: {i['full_name']} | Stars: {i['stargazers_count']} | URL: {i['html_url']}\nDesc: {i['description']}" for i in items]
                return ToolResult(success=True, output="\n---\n".join(output) if output else "No repositories found.")
                
            elif action == "get_file":
                if not repo or not path:
                    return ToolResult(success=False, output=None, error="Missing 'repo' or 'path' for get_file.")
                    
                url = f"{base_url}/repos/{repo}/contents/{path}"
                res = requests.get(url, headers=headers, timeout=10)
                res.raise_for_status()
                
                data = res.json()
                if data.get("type") == "file" and data.get("encoding") == "base64":
                    content = base64.b64decode(data["content"]).decode("utf-8")
                    return ToolResult(success=True, output=content[:20000]) # Cap output length
                return ToolResult(success=False, output=None, error="Target is not a file or cannot be decoded.")
                
            elif action == "get_issue":
                if not repo or not issue_number:
                    return ToolResult(success=False, output=None, error="Missing 'repo' or 'issue_number' for get_issue.")
                    
                url = f"{base_url}/repos/{repo}/issues/{issue_number}"
                res = requests.get(url, headers=headers, timeout=10)
                res.raise_for_status()
                
                data = res.json()
                output = f"Title: {data.get('title')}\nState: {data.get('state')}\nBody:\n{data.get('body')}"
                return ToolResult(success=True, output=output)
                
            elif action == "list_commits":
                if not repo:
                    return ToolResult(success=False, output=None, error="Missing 'repo' for list_commits.")
                    
                url = f"{base_url}/repos/{repo}/commits"
                res = requests.get(url, headers=headers, params={"per_page": 5}, timeout=10)
                res.raise_for_status()
                
                commits = res.json()
                output = [f"Commit: {c['sha'][:7]} by {c['commit']['author']['name']}\nMessage: {c['commit']['message'].splitlines()[0]}" for c in commits]
                return ToolResult(success=True, output="\n\n".join(output))
                
            else:
                return ToolResult(success=False, output=None, error=f"Unknown action: {action}")
                
        except requests.exceptions.RequestException as e:
            if hasattr(e, 'response') and e.response is not None:
                return ToolResult(success=False, output=None, error=f"GitHub API Error: {e.response.status_code} - {e.response.text}")
            return ToolResult(success=False, output=None, error=f"Request failed: {str(e)}")
        except Exception as e:
            return ToolResult(success=False, output=None, error=f"Failed to execute GitHub action: {str(e)}")

