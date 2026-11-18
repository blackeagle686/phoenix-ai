import asyncio
import inspect
import time
from typing import Any, Callable, Dict, List, Optional
from phoenix.framework.agent.tools.registry import ToolRegistry
from phoenix.framework.agent.tools.base import BaseTool, ToolResult, _normalize_tool_output
from phoenix.services.observability.tracing import tracer


class ToolManager:
    """Manages tool execution, access policies, hooks, and safety."""

    def __init__(
        self,
        registry: ToolRegistry,
        profile: Optional[Any] = None,
        runtime: Optional[Any] = None,
        before_tool_hooks: Optional[List[Callable]] = None,
        after_tool_hooks: Optional[List[Callable]] = None,
    ):
        self.registry = registry
        self.profile = profile
        self.runtime = runtime
        if runtime is not None and isinstance(self.registry, ToolRegistry):
            self.registry.set_runtime(runtime)
        self.before_tool_hooks: List[Callable] = list(before_tool_hooks or [])
        self.after_tool_hooks: List[Callable] = list(after_tool_hooks or [])

    def add_before_hook(self, hook: Callable) -> None:
        self.before_tool_hooks.append(hook)

    def add_after_hook(self, hook: Callable) -> None:
        self.after_tool_hooks.append(hook)

    register_before_hook = add_before_hook
    register_after_hook = add_after_hook

    def register_tool(self, tool_or_func: Any, **kwargs: Any) -> BaseTool:
        """Register a custom tool or function into the underlying ToolRegistry."""
        return self.registry.register(tool_or_func, **kwargs)

    def get_tool(self, tool_name: str) -> Optional[Any]:
        try:
            return self.registry.get_tool(tool_name)
        except Exception:
            return None

    def is_tool_allowed(self, tool_name: str) -> bool:
        if not self.profile or not getattr(self.profile, "tool_access", None):
            return True
        allowed = list(self.profile.tool_access)
        if "*" in allowed or tool_name in allowed:
            return True
        return False

    def get_available_tools_info(self) -> List[Dict[str, Any]]:
        """Return metadata and parameter schemas for all permitted tools in the registry."""
        if not self.registry or not hasattr(self.registry, "get_all_tools_info"):
            return []
        all_info = self.registry.get_all_tools_info()
        if not isinstance(all_info, list):
            return []
        return [
            info for info in all_info
            if isinstance(info, dict) and self.is_tool_allowed(info.get("name", ""))
        ]

    def get_openai_tools(self) -> List[Dict[str, Any]]:
        """Return OpenAI-format `tools` schemas filtered by agent profile access."""
        if not self.registry:
            return []
        if hasattr(self.registry, "to_openai_tools"):
            raw = self.registry.to_openai_tools()
            if not isinstance(raw, list):
                return []
            return [
                item for item in raw
                if isinstance(item, dict)
                and self.is_tool_allowed(item.get("function", {}).get("name", ""))
            ]
        return [
            {
                "type": "function",
                "function": {
                    "name": info.get("name", ""),
                    "description": info.get("description", ""),
                    "parameters": info.get("parameters", {"type": "object", "properties": {}}),
                },
            }
            for info in self.get_available_tools_info()
        ]

    async def _call_tool_instance(self, tool: Any, payload: Dict[str, Any]) -> ToolResult:
        if isinstance(tool, BaseTool):
            return await tool.invoke(**payload)
        exec_fn = getattr(tool, "execute", None)
        if exec_fn is None and callable(tool):
            exec_fn = tool
        if exec_fn is None:
            return ToolResult(success=False, output=None, error=f"Tool '{tool}' is not executable.")
        if inspect.iscoroutinefunction(exec_fn):
            raw = await exec_fn(**payload)
        else:
            res = exec_fn(**payload)
            raw = await res if inspect.isawaitable(res) else res
        return _normalize_tool_output(raw)

    async def execute_tool_result(
        self,
        tool_name: str,
        kwargs: Optional[Dict[str, Any]] = None,
        **extra_kwargs: Any,
    ) -> ToolResult:
        payload = dict(kwargs or {})
        if extra_kwargs:
            payload.update(extra_kwargs)

        if not self.is_tool_allowed(tool_name):
            return ToolResult(
                success=False,
                output=None,
                error=f"Permission denied: Tool '{tool_name}' is not permitted by the agent profile.",
            )

        for hook in self.before_tool_hooks:
            hook_res = hook(tool_name, payload)
            if inspect.isawaitable(hook_res):
                hook_res = await hook_res
            if hook_res is False:
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Execution of tool '{tool_name}' was rejected by a before-tool hook.",
                )
            if isinstance(hook_res, ToolResult):
                return hook_res

        t0 = time.perf_counter()
        try:
            tool = self.registry.get_tool(tool_name)
            result = await self._call_tool_instance(tool, payload)
        except ValueError as e:
            result = ToolResult(success=False, output=None, error=str(e))
        except Exception as e:
            result = ToolResult(success=False, output=None, error=f"Unexpected error during tool execution: {e}")

        elapsed_ms = (time.perf_counter() - t0) * 1000
        tracer.record_tool(tool_name, elapsed_ms, bool(getattr(result, "success", False)))

        for hook in self.after_tool_hooks:
            hook_res = hook(tool_name, payload, result)
            if inspect.isawaitable(hook_res):
                hook_res = await hook_res
            if isinstance(hook_res, ToolResult):
                result = hook_res

        return result

    async def execute_tool(self, tool_name: str, kwargs: Optional[dict] = None, **extra_kwargs: Any) -> str:
        payload = dict(kwargs or {})
        if extra_kwargs:
            payload.update(extra_kwargs)

        try:
            tool = self.registry.get_tool(tool_name)
        except ValueError as e:
            return str(e)
        except Exception as e:
            return f"Unexpected error during tool execution: {e}"

        if not self.is_tool_allowed(tool_name):
            return f"Tool execution failed: Permission denied for '{tool_name}'"

        try:
            result = await self._call_tool_instance(tool, payload)
            if result.success:
                output = result.output
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
