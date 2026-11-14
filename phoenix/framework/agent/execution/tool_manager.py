import inspect
from typing import Any, Callable, Dict, List, Optional
from phoenix.framework.agent.tools.registry import ToolRegistry
from phoenix.framework.agent.tools.base import ToolResult


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
        if runtime is not None and hasattr(self.registry, "set_runtime"):
            self.registry.set_runtime(runtime)
        self.before_tool_hooks: List[Callable] = list(before_tool_hooks or [])
        self.after_tool_hooks: List[Callable] = list(after_tool_hooks or [])

    def add_before_hook(self, hook: Callable) -> None:
        self.before_tool_hooks.append(hook)

    def add_after_hook(self, hook: Callable) -> None:
        self.after_tool_hooks.append(hook)

    register_before_hook = add_before_hook
    register_after_hook = add_after_hook

    def is_tool_allowed(self, tool_name: str) -> bool:
        if not self.profile or not getattr(self.profile, "tool_access", None):
            return True
        allowed = list(self.profile.tool_access)
        if "*" in allowed or tool_name in allowed:
            return True
        return False

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

        try:
            tool = self.registry.get_tool(tool_name)
            result = await tool.execute(**payload)
            if not isinstance(result, ToolResult):
                result = ToolResult(success=True, output=result)
        except ValueError as e:
            result = ToolResult(success=False, output=None, error=str(e))
        except Exception as e:
            result = ToolResult(success=False, output=None, error=f"Unexpected error during tool execution: {e}")

        for hook in self.after_tool_hooks:
            hook_res = hook(tool_name, payload, result)
            if inspect.isawaitable(hook_res):
                hook_res = await hook_res
            if isinstance(hook_res, ToolResult):
                result = hook_res

        return result

    async def execute_tool(self, tool_name: str, kwargs: dict) -> str:
        try:
            tool = self.registry.get_tool(tool_name)
        except ValueError as e:
            return str(e)
        except Exception as e:
            return f"Unexpected error during tool execution: {e}"

        if not self.is_tool_allowed(tool_name):
            return f"Tool execution failed: Permission denied for '{tool_name}'"

        try:
            result = await tool.execute(**(kwargs or {}))
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
