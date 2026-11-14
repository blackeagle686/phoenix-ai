import inspect
from typing import Any, Callable, Dict, Optional, get_origin, get_args, Union
from pydantic import BaseModel


class ToolResult(BaseModel):
    success: bool
    output: Any
    error: Optional[str] = None


def _python_type_to_json_type(annotation: Any) -> str:
    if annotation is inspect.Parameter.empty or annotation is Any:
        return "string"
    origin = get_origin(annotation)
    if origin is Union:
        non_none = [a for a in get_args(annotation) if a is not type(None)]
        if non_none:
            return _python_type_to_json_type(non_none[0])
    if origin in (list, tuple, set):
        return "array"
    if origin is dict:
        return "object"
    if annotation in (int,):
        return "integer"
    if annotation in (float,):
        return "number"
    if annotation in (bool,):
        return "boolean"
    if annotation in (list, tuple, set):
        return "array"
    if annotation in (dict,):
        return "object"
    return "string"


def _extract_parameters_schema(fn: Callable) -> Dict[str, Any]:
    try:
        sig = inspect.signature(fn)
    except (ValueError, TypeError):
        return {"type": "object", "properties": {}, "required": []}

    properties: Dict[str, Any] = {}
    required = []

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue

        json_type = _python_type_to_json_type(param.annotation)
        prop_info: Dict[str, Any] = {"type": json_type}
        if param.default is not inspect.Parameter.empty:
            if param.default is not None:
                prop_info["default"] = param.default
        else:
            required.append(param_name)
        properties[param_name] = prop_info

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


class BaseTool:
    name: str
    description: str

    async def execute(self, **kwargs) -> ToolResult:
        raise NotImplementedError("Tool must implement execute method")

    async def run(self, **kwargs) -> Any:
        result = await self.execute(**kwargs)
        return result.output if result.success else result.error

    def get_parameters_schema(self) -> Dict[str, Any]:
        target_fn = getattr(self, "func", self.execute)
        return _extract_parameters_schema(target_fn)

    def schema(self) -> Dict[str, Any]:
        return self.to_dict()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.get_parameters_schema(),
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
                    if inspect.iscoroutinefunction(self.func):
                        result = await self.func(**kwargs)
                    else:
                        result = self.func(**kwargs)
                    return ToolResult(success=True, output=result)
                except Exception as e:
                    return ToolResult(success=False, output="", error=str(e))

        return FunctionTool()

    return decorator
