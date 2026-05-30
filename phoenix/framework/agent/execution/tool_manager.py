from phoenix.framework.agent.tools.registry import ToolRegistry

class ToolManager:
    """Manages tool execution and safety."""
    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    async def execute_tool(self, tool_name: str, kwargs: dict) -> str:
        try:
            tool = self.registry.get_tool(tool_name)
            result = await tool.execute(**kwargs)
            if result.success:
                output = result.output
                # If a tool returns chunked output (list of strings), format into a readable chunked string
                if isinstance(output, list):
                    try:
                        parts = []
                        total = len(output)
                        for i, chunk in enumerate(output, start=1):
                            parts.append(f"---FILE CHUNK {i}/{total}---\n{chunk}")
                        return "\n\n".join(parts)
                    except Exception:
                        return str(output)
                return str(output)
            else:
                return f"Tool execution failed: {result.error}"
        except ValueError as e:
            return str(e)
        except Exception as e:
            return f"Unexpected error during tool execution: {e}"

