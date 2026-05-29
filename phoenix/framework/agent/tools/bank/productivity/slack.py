import os
import json
import requests
from typing import Optional
from phoenix.framework.agent.tools.base import BaseTool, ToolResult

class SlackTool(BaseTool):
    name = "slack"
    description = (
        "Sends a message to a Slack channel via an Incoming Webhook. "
        "Input: 'message' (str, the message to send)."
    )

    async def execute(self, message: str, webhook_url: Optional[str] = None, **kwargs) -> ToolResult:
        try:
            url = webhook_url or os.environ.get("SLACK_WEBHOOK_URL")
            
            if not url:
                return ToolResult(
                    success=False, 
                    output=None, 
                    error="Slack Webhook URL is missing. Provide it in arguments or set SLACK_WEBHOOK_URL environment variable."
                )

            headers = {"Content-type": "application/json"}
            payload = {"text": message}
            
            response = requests.post(url, data=json.dumps(payload), headers=headers, timeout=10)
            response.raise_for_status()
            
            return ToolResult(success=True, output="Message sent to Slack successfully.")
            
        except requests.exceptions.RequestException as e:
            return ToolResult(success=False, output=None, error=f"Failed to send Slack message: {str(e)}")
        except Exception as e:
            return ToolResult(success=False, output=None, error=f"An error occurred while sending to Slack: {str(e)}")
