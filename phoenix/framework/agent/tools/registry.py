from typing import Any, Callable, Dict, Iterable, List, Optional
from phoenix.framework.agent.tools.base import BaseTool, ensure_tool, create_tool
from phoenix.framework.agent.tools.search import WebSearchTool
from phoenix.framework.agent.tools.code import (
    CodeExecutionTool,
    PythonAnalyzerTool,
    CommandExecutionTool,
    CodeCompileTool,
)
from phoenix.framework.agent.tools.io import (
    FileReadTool,
    FileWriteTool,
    FileSearchTool,
    FileAppendTool,
    FileEditTool,
    FileDeleteTool,
)
from phoenix.framework.agent.tools.patch import MultiBlockUpdateTool


class ToolRegistry:
    def __init__(self, runtime: Optional[Any] = None, tools: Optional[Iterable[Any]] = None):
        self.tools: Dict[str, BaseTool] = {}
        self.runtime = runtime
        if tools:
            self.register_many(tools)

    def register(
        self,
        tool_or_func: Any,
        name: Optional[str] = None,
        description: Optional[str] = None,
        **kwargs: Any,
    ) -> BaseTool:
        """
        Register a tool in the registry.
        Accepts a `BaseTool` instance, a `BaseTool` subclass, a `@tool`-decorated function,
        a plain Python function (sync or async), or a dict `{"func": ..., "name": ..., "description": ...}`.
        """
        tool_obj = ensure_tool(tool_or_func, name=name, description=description, **kwargs)
        if self.runtime is not None:
            should_bind = (
                getattr(tool_obj, "runtime", None) is None
                or getattr(tool_obj, "_default_runtime_created", False)
            )
            if should_bind:
                if hasattr(tool_obj, "set_runtime"):
                    tool_obj.set_runtime(self.runtime)
                elif hasattr(tool_obj, "runtime"):
                    tool_obj.runtime = self.runtime
        self.tools[tool_obj.name] = tool_obj
        return tool_obj

    def register_many(self, tools: Iterable[Any]) -> List[BaseTool]:
        """Register multiple tools or functions at once."""
        return [self.register(t) for t in tools]

    def register_function(
        self,
        func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None,
        **kwargs: Any,
    ) -> BaseTool:
        """Explicitly wrap and register a Python callable as a tool."""
        tool_obj = create_tool(func, name=name, description=description, **kwargs)
        return self.register(tool_obj)

    def unregister(self, name: str) -> bool:
        """Remove a tool by name from the registry."""
        if name in self.tools:
            del self.tools[name]
            return True
        return False

    def set_runtime(self, runtime: Any):
        self.runtime = runtime
        for tool_obj in self.tools.values():
            if hasattr(tool_obj, "set_runtime"):
                tool_obj.set_runtime(runtime)
            elif hasattr(tool_obj, "runtime"):
                tool_obj.runtime = runtime

    def get_tool(self, name: str) -> BaseTool:
        if name not in self.tools:
            available = ", ".join(self.tools.keys()) or "none"
            raise ValueError(f"Tool '{name}' not found in registry. Available tools: [{available}]")
        return self.tools[name]

    def has_tool(self, name: str) -> bool:
        return name in self.tools

    def list_tools(self) -> List[str]:
        return list(self.tools.keys())

    def get_all_tools_info(self) -> List[Dict]:
        return [t.to_dict() for t in self.tools.values()]

    def to_openai_tools(self) -> List[Dict[str, Any]]:
        return [
            t.to_openai_tool()
            if hasattr(t, "to_openai_tool")
            else {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": getattr(t, "description", ""),
                    "parameters": t.get_parameters_schema() if hasattr(t, "get_parameters_schema") else {"type": "object", "properties": {}},
                },
            }
            for t in self.tools.values()
        ]

    @classmethod
    def load_default(
        cls,
        runtime: Optional[Any] = None,
        extra_tools: Optional[Iterable[Any]] = None,
    ) -> "ToolRegistry":
        registry = cls(runtime=runtime)
        registry.register(WebSearchTool())
        registry.register(CodeExecutionTool(runtime=runtime))
        registry.register(PythonAnalyzerTool(runtime=runtime))
        registry.register(CommandExecutionTool(runtime=runtime))
        registry.register(CodeCompileTool(runtime=runtime))
        registry.register(FileReadTool(runtime=runtime))
        registry.register(FileWriteTool(runtime=runtime))
        registry.register(FileAppendTool(runtime=runtime))
        registry.register(FileEditTool(runtime=runtime))
        registry.register(FileSearchTool(runtime=runtime))
        registry.register(FileDeleteTool(runtime=runtime))
        registry.register(MultiBlockUpdateTool(runtime=runtime))
        if extra_tools:
            registry.register_many(extra_tools)
        return registry
