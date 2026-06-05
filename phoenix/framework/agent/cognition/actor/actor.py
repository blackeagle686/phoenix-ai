import json
from typing import Optional, Any
from uuid import uuid4
from .base import BaseActor
from phoenix.framework.agent.tools.base import ToolResult
from phoenix.framework.agent.cognition.actor.schema import ActorInputSchema, ActorOutputSchema, ActorToReflectorSchema
from phoenix.framework.agent.cognition.planner.schema import Task, TaskType
from phoenix.framework.agent.cognition.utils import parse_llm_json

class Actor(BaseActor):
    """
    Strict execution engine that binds the Planner's cognitive tasks to real-world Tools.
    Equipped with bi-directional Reflector bindings for real-time execution evaluation.
    """
    
    def __init__(self, tool_manager, llm: Optional[Any] = None, reflector: Optional[Any] = None):
        super().__init__(tool_manager, llm=llm, reflector=reflector)
        
    async def generate_and_execute(self, task: Task, previous_results: str = "") -> ActorOutputSchema:
        """
        Dynamically generates the required code and tool payload based on the Task's high-level strategy,
        then executes the tool immediately.
        """
        if not self.llm:
            raise ValueError("Actor requires an LLM instance to generate actions from a Task.")
            
        solution_context = task.payload.get("solution_context", "No context provided.")
        tools_required = task.payload.get("tools_required", [])
        
        # Build the prompt
        prompt = f"""
        You are the Execution Engine of an autonomous AI agent.
        Your goal is to implement the next step for the following task.
        
        Task Description: {task.description}
        Architectural Plan / Strategy: {solution_context}
        Previous Results: {previous_results if previous_results else 'None so far.'}
        
        Available Tools: {json.dumps(tools_required)}
        
        Based on the plan and previous results, generate the exact code or payload for the NEXT single step.
        If the task requires creating a file, generate the FULL source code and use the 'file_write' tool.
        If the task requires a directory, use 'folder_create'.
        If the plan is complete or no more actions are needed, use the tool 'finish'.
        
        Respond ONLY in valid JSON matching this exact structure:
        {{
            "tool_name": "name_of_the_tool_to_use",
            "payload": {{
                "argument_name": "argument_value"
            }}
        }}
        """
        
        response = await self.llm.generate(prompt, max_tokens=4000)
        data = parse_llm_json(response) or {}
        
        tool_name = data.get("tool_name", "finish")
        payload = data.get("payload", {})
        
        # If the LLM decides it's finished, return a pseudo-success
        if tool_name == "finish":
            return ActorOutputSchema(
                task_id=task.task_id,
                success=True,
                result={"status": "finished", "message": "Actor determined no further actions are needed."},
                error_context=None
            )
            
        # Construct the strict ActorInputSchema
        task_input = ActorInputSchema(
            task_id=task.task_id,
            task_type=TaskType.OTHER,
            tool_name=tool_name,
            payload=payload
        )
        
        # Delegate to standard execution
        return await self.execute(task_input, task_context=task)

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

if __name__ == "__main__":
    import asyncio
    from phoenix.services.llm.openai import OpenAILLM
    from phoenix.framework.agent.execution.tool_manager import ToolManager
    from phoenix.framework.agent.tools.registry import ToolRegistry
    
    llm = OpenAILLM(
        api_key="ak_2yp3Xw1Ny7ky2pF7er9x93ZO9jj6G",
        model="LongCat-2.0-Preview",
        base_url="https://api.longcat.chat/openai"
    )

    class MockTool:
        def __init__(self, name):
            self.name = name
            
        async def execute(self, **kwargs):
            return ToolResult(success=True, output=f"Executed {self.name} with payload: {kwargs}", error=None)

    async def run_test():
        print("Initializing LLM...", flush=True)
        await llm.init()
        print("LLM Initialized.", flush=True)
        
        registry = ToolRegistry()
        registry.register(MockTool("file_write"))
        registry.register(MockTool("folder_create"))
        
        tool_manager = ToolManager(registry)
        actor = Actor(tool_manager=tool_manager, llm=llm)
        
        # Create a mock Task
        from phoenix.framework.agent.cognition.reflector.schema import BaseReflectorMeta
        mock_task = Task(
            prompt_id=uuid4(),
            task_id=str(uuid4()),
            dependencies=[],
            task_type=TaskType.WRITE,
            task_title="Test Code Gen",
            description="Create the JWT middleware module for the Actix API.",
            payload={
                "tools_required": ["file_write", "folder_create"],
                "solution_context": "- Implement JWT-based middleware using jsonwebtoken crate."
            },
            status="pending",
            priority="medium",
            complexity="medium",
            created_by="Test",
            problems=[],
            reflector_result=BaseReflectorMeta(rating=5, feedback="Mock", confidence=1.0, reasoning="Test")
        )
        
        print("Starting Actor.generate_and_execute test...")
        output = await actor.generate_and_execute(mock_task)
        
        print("\n" + "="*60)
        print("ACTOR EXECUTION RESULT")
        print("="*60)
        print(output.model_dump_json(indent=5))
        print("="*60 + "\n")

    asyncio.run(run_test())
