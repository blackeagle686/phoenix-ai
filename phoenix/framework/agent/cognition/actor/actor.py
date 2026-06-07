from typing import Optional, Any
from .base import BaseActor
from phoenix.framework.agent.tools.base import ToolResult
from phoenix.framework.agent.cognition.actor.schema import ActorInputSchema, ActorOutputSchema, ActorToReflectorSchema
from phoenix.framework.agent.cognition.schemas.brain import ActionSchema, SolutionSchema, ProblemSchema

class Actor(BaseActor):
    """
    Strict execution engine that binds the Thinker's cognitive outputs to real-world Tools
    via the Runtime. Does not perform LLM calls itself.
    """
    
    def __init__(self, tool_manager, thinker, runtime=None, reflector: Optional[Any] = None):
        super().__init__(tool_manager=tool_manager, thinker=thinker, reflector=reflector, runtime=runtime)
        
    async def generate_and_execute(self, task: Any, previous_results: str = "") -> ActorOutputSchema:
        """
        Dynamically generates the required code and tool payload based on the Task's high-level strategy
        by delegating strictly to the Thinker, then executes the tool via Runtime.
        """
        if not self.thinker:
            raise ValueError("Actor requires a Thinker instance to generate actions from a Task.")
            
        # 1. Ask Thinker to create solutions for the task/problem
        # Assuming the Planner has already defined problems and attached them to the task,
        # but if not, we can ask Thinker directly or assume task is a ProblemSchema.
        # For this refactor, we directly get the ActionSchema from the Thinker based on the task.
        
        # We ask Thinker to define solutions for this task.
        solution = await self.thinker.create_solutions(task)
        
        # 2. Ask Thinker to generate strict action payload (tools, code, I/O paths)
        action_payload: ActionSchema = await self.thinker.generate_action_payload(solution)
        
        results = []
        success = True
        
        # 3. Execute IO Operations via Runtime
        if self.runtime and hasattr(self.runtime, "execute_io"):
            for io_op in action_payload.io_operations:
                res = await self.runtime.execute_io(io_op.operation, io_op.file_path, getattr(io_op, "content", None))
                results.append({"io_op": io_op.operation, "path": io_op.file_path, "success": res.success, "output": res.output, "error": res.error})
                if not res.success: success = False
        else:
            if action_payload.io_operations:
                results.append({"io_op": "all", "success": False, "error": "Runtime does not support execute_io."})
                success = False

        # 4. Execute Tools STRICTLY via Runtime
        for tool_call in action_payload.tools_to_call:
            try:
                tool_name = tool_call.tool_name
                payload = tool_call.arguments
                
                if tool_name in ["python_repl", "execute_code", "python"] and self.runtime and hasattr(self.runtime, "execute_code"):
                    code = payload.get("code", "")
                    res = await self.runtime.execute_code(code)
                    results.append({"tool": tool_name, "success": res.success, "output": res.output, "error": getattr(res, "error", None)})
                    if not res.success: success = False
                elif tool_name in ["execute_command", "bash"] and self.runtime and hasattr(self.runtime, "execute_command"):
                    cmd = payload.get("command", "")
                    cwd = payload.get("cwd")
                    res = await self.runtime.execute_command(cmd, cwd=cwd)
                    results.append({"tool": tool_name, "success": res.success, "output": res.output, "error": getattr(res, "error", None)})
                    if not res.success: success = False
                elif tool_name in ["file_read", "file_write", "file_edit", "file_append", "file_delete"] and self.runtime and hasattr(self.runtime, "execute_io"):
                    op_map = {
                        "file_read": "read", "file_write": "create", 
                        "file_edit": "edit", "file_append": "append", "file_delete": "delete"
                    }
                    op = op_map.get(tool_name)
                    path = payload.get("file_path", payload.get("path", ""))
                    content = payload.get("content", payload.get("write_content", payload.get("edit_content", "")))
                    res = await self.runtime.execute_io(op, path, content)
                    results.append({"tool": tool_name, "success": res.success, "output": res.output, "error": getattr(res, "error", None)})
                    if not res.success: success = False
                else:
                    # Enforce strict boundary: No tool manager fallback.
                    results.append({"tool": tool_name, "success": False, "error": f"Tool '{tool_name}' blocked. Strict runtime execution enforced."})
                    success = False
            except Exception as e:
                results.append({"tool": tool_call.tool_name, "success": False, "error": str(e)})
                success = False

        # 5. Return Output
        task_id = getattr(task, "task_id", "unknown")
        actor_output = ActorOutputSchema(
            task_id=task_id,
            tool_name="multiple_actions",
            success=success,
            result={"execution_results": results, "action_plan": action_payload.action_plan},
            error_context="Some tools failed" if not success else None
        )
        
        # Note: Reflector execution is handled by the loop/reflector worker in the new architecture,
        # so we don't necessarily need to trigger it here unless keeping backward compatibility.
        
        return actor_output

    async def execute(self, task_input: ActorInputSchema, task_context: Optional[Any] = None) -> ActorOutputSchema:
        """Legacy strict execution bypass if needed."""
        pass
