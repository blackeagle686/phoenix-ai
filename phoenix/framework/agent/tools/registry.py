from typing import Any, Dict, List, Optional
from phoenix.framework.agent.tools.base import BaseTool
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
    def __init__(self, runtime: Optional[Any] = None):
        self.tools: Dict[str, BaseTool] = {}
        self.runtime = runtime

    def register(self, tool: BaseTool):
        if self.runtime is not None and hasattr(tool, "runtime"):
            tool.runtime = self.runtime
        self.tools[tool.name] = tool

    def set_runtime(self, runtime: Any):
        self.runtime = runtime
        for tool in self.tools.values():
            if hasattr(tool, "runtime"):
                tool.runtime = runtime

    def get_tool(self, name: str) -> BaseTool:
        if name not in self.tools:
            raise ValueError(f"Tool {name} not found in registry")
        return self.tools[name]

    def has_tool(self, name: str) -> bool:
        return name in self.tools

    def get_all_tools_info(self) -> List[Dict]:
        return [t.to_dict() for t in self.tools.values()]

    @classmethod
    def load_default(cls, runtime: Optional[Any] = None) -> "ToolRegistry":
        registry = cls(runtime=runtime)
        registry.register(WebSearchTool())
        registry.register(CodeExecutionTool(runtime=runtime))
        registry.register(PythonAnalyzerTool())
        registry.register(CommandExecutionTool(runtime=runtime))
        registry.register(CodeCompileTool())
        registry.register(FileReadTool(runtime=runtime))
        registry.register(FileWriteTool(runtime=runtime))
        registry.register(FileAppendTool(runtime=runtime))
        registry.register(FileEditTool(runtime=runtime))
        registry.register(FileSearchTool(runtime=runtime))
        registry.register(FileDeleteTool(runtime=runtime))
        registry.register(MultiBlockUpdateTool(runtime=runtime))
        return registry
