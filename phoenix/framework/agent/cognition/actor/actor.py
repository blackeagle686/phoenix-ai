from __future__ import annotations
from typing import Optional, Any, TYPE_CHECKING
from .base import BaseActor
from phoenix.framework.agent.tools.base import ToolResult
from phoenix.framework.agent.cognition.actor.schema import ActorInputSchema, ActorOutputSchema, ActorToReflectorSchema

if TYPE_CHECKING:
    from phoenix.framework.agent.cognition.schemas.brain import ActionSchema, SolutionSchema, ProblemSchema


class Actor(BaseActor):
    """Strict execution engine that binds Thinker outputs to Tools via Runtime"""

    def __init__(self, tool_manager, thinker, runtime=None, reflector: Optional[Any] = None):
        super().__init__(tool_manager=tool_manager, thinker=thinker, reflector=reflector, runtime=runtime)

    async def generate_and_execute(self, task: Any, previous_results: str = "", context: str = "") -> ActorOutputSchema:
        """Generate actions from Thinker then execute tools and IO operations"""

        if not self.thinker:
            raise ValueError("Actor requires a Thinker instance to generate actions from a Task.")

        solution = await self.thinker.create_solutions(task, context=context)
        action_payload: ActionSchema = await self.thinker.generate_action_payload(solution, context=context)

        results = []
        success = True

        if self.runtime and hasattr(self.runtime, "execute_io"):
            for io_op in action_payload.io_operations:
                try:
                    res = await self.runtime.execute_io(io_op.operation, io_op.file_path, getattr(io_op, "content", None))
                    results.append({
                        "io_op": io_op.operation,
                        "path": io_op.file_path,
                        "success": res.success,
                        "output": res.output,
                        "error": res.error
                    })
                    if not res.success:
                        success = False
                except Exception as e:
                    results.append({
                        "io_op": io_op.operation,
                        "path": io_op.file_path,
                        "success": False,
                        "output": None,
                        "error": str(e)
                    })
                    success = False
        else:
            if action_payload.io_operations:
                for io_op in action_payload.io_operations:
                    results.append({
                        "io_op": io_op.operation,
                        "path": io_op.file_path,
                        "success": False,
                        "error": "Runtime does not support execute_io."
                    })
                success = False

        for tool_call in action_payload.tools_to_call:
            try:
                tool_name = tool_call.tool_name
                payload = tool_call.arguments

                if tool_name in ["python_repl", "execute_code", "python"] and self.runtime and hasattr(self.runtime, "execute_code"):
                    code = payload.get("code", "")
                    res = await self.runtime.execute_code(code)
                    results.append({
                        "tool": tool_name,
                        "arguments": payload,
                        "success": res.success,
                        "output": res.output,
                        "error": getattr(res, "error", None)
                    })
                    if not res.success:
                        success = False

                elif tool_name in ["execute_command", "bash"] and self.runtime and hasattr(self.runtime, "execute_command"):
                    cmd = payload.get("command", "")
                    cwd = payload.get("cwd")
                    res = await self.runtime.execute_command(cmd, cwd=cwd)
                    results.append({
                        "tool": tool_name,
                        "arguments": payload,
                        "success": res.success,
                        "output": res.output,
                        "error": getattr(res, "error", None)
                    })
                    if not res.success:
                        success = False

                elif tool_name in ["file_read", "file_write", "file_edit", "file_append", "file_delete"] and self.runtime and hasattr(self.runtime, "execute_io"):
                    op_map = {
                        "file_read": "read", "file_write": "create",
                        "file_edit": "edit", "file_append": "append", "file_delete": "delete"
                    }
                    op = op_map.get(tool_name)
                    path = payload.get("file_path", payload.get("path", ""))
                    content = payload.get("content", payload.get("write_content", payload.get("edit_content", "")))
                    res = await self.runtime.execute_io(op, path, content)
                    results.append({
                        "tool": tool_name,
                        "io_op": op,
                        "path": path,
                        "arguments": payload,
                        "success": res.success,
                        "output": res.output,
                        "error": getattr(res, "error", None)
                    })
                    if not res.success:
                        success = False

                else:
                    results.append({
                        "tool": tool_name,
                        "arguments": payload,
                        "success": False,
                        "error": f"Tool '{tool_name}' blocked. Strict runtime execution enforced."
                    })
                    success = False

            except Exception as e:
                results.append({
                    "tool": tool_call.tool_name,
                    "arguments": tool_call.arguments,
                    "success": False,
                    "error": str(e)
                })
                success = False

        task_id = getattr(task, "task_id", "unknown")
        actor_output = ActorOutputSchema(
            task_id=task_id,
            tool_name="multiple_actions",
            success=success,
            result={"execution_results": results, "action_plan": action_payload.action_plan},
            error_context="Some tools failed" if not success else None
        )

        return actor_output

    async def execute(self, task_input: ActorInputSchema, task_context: Optional[Any] = None) -> ActorOutputSchema:
        """Legacy strict execution bypass"""
        pass
