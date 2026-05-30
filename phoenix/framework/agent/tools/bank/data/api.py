import httpx
from typing import Optional, Dict, Any
from phoenix.framework.agent.tools.base import BaseTool, ToolResult

class APIRESTTool(BaseTool):
    name = "api_rest"
    description = (
        "Interacts with standard REST APIs via HTTP requests. "
        "Inputs: 'method' (str, e.g., 'GET', 'POST', 'PUT', 'DELETE'), "
        "'url' (str, the fully qualified API endpoint URL), "
        "'headers' (dict, optional HTTP headers, e.g., {'Authorization': 'Bearer token'}), "
        "'json_body' (dict, optional JSON payload for POST/PUT), "
        "'params' (dict, optional URL query parameters)."
    )

    async def execute(
        self, 
        method: str, 
        url: str, 
        headers: Optional[Dict[str, str]] = None, 
        json_body: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> ToolResult:
        method = method.upper()
        if method not in {"GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"}:
            return ToolResult(success=False, output=None, error=f"Unsupported HTTP method: {method}")

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.request(
                    method=method,
                    url=url,
                    headers=headers,
                    json=json_body,
                    params=params
                )
                
                # Attempt to parse as JSON for pretty formatting
                try:
                    data = response.json()
                    import json
                    output = json.dumps(data, indent=2)
                except ValueError:
                    output = response.text
                
                output_str = f"Status Code: {response.status_code}\nResponse:\n{output}"
                
                if response.is_error:
                    return ToolResult(success=False, output=None, error=f"HTTP API Error: \n{output_str[:5000]}")
                    
                return ToolResult(success=True, output=output_str[:15000]) # Cap to prevent overwhelming the LLM
                
        except httpx.RequestError as e:
            return ToolResult(success=False, output=None, error=f"Request failed: {str(e)}")
        except Exception as e:
            return ToolResult(success=False, output=None, error=f"API Execution Error: {str(e)}")

