from typing import Dict, List, Type
from phoenix.framework.agent.tools.base import BaseTool
from phoenix.framework.agent.tools.search import WebSearchTool
from phoenix.framework.agent.tools.code import CodeExecutionTool, PythonAnalyzerTool
from phoenix.framework.agent.tools.io import FileReadTool, FileWriteTool, FileSearchTool, FileAppendTool, FileEditTool, FileDeleteTool
from phoenix.framework.agent.tools.patch import MultiBlockUpdateTool

class ToolRegistry:
    def __init__(self):
        self.tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool):
        self.tools[tool.name] = tool

    def get_tool(self, name: str) -> BaseTool:
        if name not in self.tools:
            raise ValueError(f"Tool {name} not found in registry")
        return self.tools[name]
        
    def get_all_tools_info(self) -> List[Dict]:
        return [t.to_dict() for t in self.tools.values()]

    @classmethod
    def load_default(cls) -> "ToolRegistry":
        registry = cls()
        registry.register(WebSearchTool())
        registry.register(CodeExecutionTool())
        registry.register(PythonAnalyzerTool())
        registry.register(FileReadTool())
        registry.register(FileWriteTool())
        registry.register(FileAppendTool())
        registry.register(FileEditTool())
        registry.register(FileSearchTool())
        registry.register(FileDeleteTool())
        registry.register(MultiBlockUpdateTool())
        return registry
