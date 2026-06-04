import json
from typing import Optional, Any
from .base import BaseActor
from phoenix.framework.agent.tools.base import ToolResult
from phoenix.framework.agent.cognition.actor.schema import ActorInputSchema, ActorOutputSchema, ActorToReflectorSchema
from phoenix.framework.agent.cognition.planner.schema import Task

class Actor(BaseActor):
    """
    Strict execution engine that binds the Planner's cognitive tasks to real-world Tools.
    Equipped with bi-directional Reflector bindings for real-time execution evaluation.
    """
    
    def __init__(self, tool_manager, reflector: Optional[Any] = None):
        super().__init__(tool_manager, reflector=reflector)
        
    async def execute(self, task_input: ActorInputSchema, task_context: Optional[Task] = None) -> ActorOutputSchema:
        """
        Executes a specific tool defined in the ActorInputSchema.
        If a Reflector is attached and task_context is provided, evaluates the outcome immediately.
        """
        tool_name = task_input.tool_name
        payload = task_input.payload
        
        # 1. Execute the Tool
        try:
            # Attempt strict ToolRegistry execution
            tool = self.tool_manager.registry.get_tool(tool_name)
            res = await tool.execute(**payload)
        except ValueError:
            # Fallback to dynamic manager if not in registry
            try:
                res_str = await self.tool_manager.execute_tool(tool_name, payload)
                success = "failed" not in res_str.lower() and "error" not in res_str.lower()
                res = ToolResult(
                    success=success,
                    output=res_str,
                    error=res_str if not success else None
                )
            except Exception as e:
                res = ToolResult(success=False, output=None, error=str(e))
        except Exception as e:
            res = ToolResult(success=False, output=None, error=str(e))

        # 2. Construct the Strict ActorOutputSchema
        output_data = None
        if res.output is not None:
            # If the output is a Pydantic Model (e.g. from FileReadResult), it's already a dict, 
            # or we can force dict if it's an object. 
            # The tools currently return .dict() directly.
            output_data = res.output if isinstance(res.output, dict) else {"raw": res.output}

        actor_output = ActorOutputSchema(
            task_id=task_input.task_id,
            success=res.success,
            result=output_data,
            error_context=res.error
        )
        
        # 3. Optional Reflection Pipeline
        if self.reflector and task_context:
            try:
                to_reflector = ActorToReflectorSchema(
                    task_context=task_context,
                    actor_output=actor_output
                )
                
                reflection_result = await self.reflector.reflect(to_reflector.to_reflector_input())
                actor_output.reflection = reflection_result
            except Exception as e:
                actor_output.error_context = (actor_output.error_context or "") + f" | Reflector Error: {str(e)}"
                
        return actor_output
