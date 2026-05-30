from typing import Any, Callable, Dict, Optional
from pydantic import BaseModel, Field

class ToolResult(BaseModel):
    success: bool
    output: Any
    error: Optional[str] = None

class BaseTool:
    name: str
    description: str
    
    async def execute(self, **kwargs) -> ToolResult:
        raise NotImplementedError("Tool must implement execute method")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description
        }

def tool(name: str, description: str):
    """
    Decorator to easily convert a python function into a BaseTool.
    """
    def decorator(func: Callable):
        class FunctionTool(BaseTool):
            def __init__(self):
                self.name = name
                self.description = description
                self.func = func

            async def execute(self, **kwargs) -> ToolResult:
                try:
                    import inspect
                    if inspect.iscoroutinefunction(self.func):
                        result = await self.func(**kwargs)
                    else:
                        result = self.func(**kwargs)
                    return ToolResult(success=True, output=result)
                except Exception as e:
                    return ToolResult(success=False, output="", error=str(e))
        return FunctionTool()
    return decorator

