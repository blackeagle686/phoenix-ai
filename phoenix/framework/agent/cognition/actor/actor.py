from __future__ import annotations
from typing import Optional, Any, TYPE_CHECKING
from .base import BaseActor
from phoenix.framework.agent.tools.base import ToolResult
from phoenix.framework.agent.cognition.schema import ActorInputSchema, ActorOutputSchema, ActorToReflectorSchema

if TYPE_CHECKING:
    from phoenix.framework.agent.cognition.schema import ActionSchema, SolutionSchema, ProblemSchema


class Actor(BaseActor):
    """Strict execution engine that binds Thinker outputs to Tools via Runtime"""

    def __init__(self, tool_manager, thinker, runtime=None, reflector: Optional[Any] = None):
        super().__init__(tool_manager=tool_manager, thinker=thinker, reflector=reflector, runtime=runtime)

    async def generate_and_execute(self, task: Any, previous_results: str = "", context: str = "") -> ActorOutputSchema:
        """Generate actions from Thinker then execute tools dynamically"""

        if not self.thinker:
            raise ValueError("Actor requires a Thinker instance to generate actions from a Task.")

        task_execution = await self.thinker.solve_task(task, context=context)

        results = []
        success = True

        for tool_call in task_execution.tools_to_call:
            try:
                tool_name = tool_call.tool_name
                payload = tool_call.arguments

                # Use the centralized ToolManager to execute the tool
                if hasattr(self.tool_manager, "registry"):
                    tool = self.tool_manager.registry.get_tool(tool_name)
                    res = await tool.execute(**payload)
                    
                    # Convert ToolResult to dict for logging
                    res_dict = res.dict() if hasattr(res, "dict") else {
                        "success": getattr(res, "success", False), 
                        "output": getattr(res, "output", str(res)),
                        "error": getattr(res, "error", None)
                    }
                    
                    results.append({
                        "tool": tool_name,
                        "arguments": payload,
                        "success": res_dict.get("success", False),
                        "output": res_dict.get("output"),
                        "error": res_dict.get("error")
                    })
                    
                    if not res_dict.get("success", False):
                        success = False
                else:
                    results.append({
                        "tool": tool_name,
                        "arguments": payload,
                        "success": False,
                        "error": "ToolManager does not have a registry."
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
            result={"execution_results": results, "thought_process": task_execution.thought_process},
            error_context="Some tools failed" if not success else None
        )

        return actor_output

    async def execute(self, task_input: ActorInputSchema, task_context: Optional[Any] = None) -> ActorOutputSchema:
        """Legacy strict execution bypass"""
        pass
