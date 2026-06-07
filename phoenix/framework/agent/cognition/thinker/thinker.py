import json
from typing import Any
from .base import BaseThinker
from ..schemas.brain import PlanSchema, ProblemSchema, SolutionSchema, ActionSchema, ReflectionSchema

class Thinker(BaseThinker):
    """
    The Central Brain. This is the only module that makes LLM generation calls.
    It takes requests from the Planner, Actor, and Reflector and returns strict schemas.
    """
    
    def __init__(self, llm, profile=None, tool_manager=None):
        super().__init__(llm, profile=profile, tool_manager=tool_manager)

    async def _execute_search_tools(self, prompt: str) -> str:
        """Helper to run search/read tools directly to gather context if needed."""
        context = ""
        if self.tool_manager:
            # Simple heuristic: if we have tool manager, we could call specific read tools.
            # For now, we will rely on the agent's memory or explicit file contents passed.
            pass
        return context

    async def generate_plan(self, prompt: str, memory: Any, session_id: str) -> PlanSchema:
        context = await memory.get_full_context(session_id, query=prompt)
        system_prompt = (
            "You are the master Thinker and Planner. Analyze the user request and project context.\n"
            "Create a strict plan containing a main objective and a list of step-by-step tasks to accomplish it."
        )
        if self.profile:
            system_prompt += f"\n\n{self.profile.to_prompt_string()}"
            
        full_prompt = f"{system_prompt}\n\nContext:\n{context}\n\nUser Request: {prompt}"
        
        # We assume the LLM implementation has `generate_structured`
        return await self.llm.generate_structured(full_prompt, PlanSchema, session_id=session_id)

    async def define_problems(self, task: Any) -> ProblemSchema:
        task_desc = getattr(task, "description", str(task))
        task_id = getattr(task, "task_id", "unknown")
        
        system_prompt = (
            "You are the Thinker. Given the following task, define the explicit problems "
            "that need to be solved. Break down the task into distinct technical problems."
        )
        full_prompt = f"{system_prompt}\n\nTask ID: {task_id}\nTask Description: {task_desc}"
        
        return await self.llm.generate_structured(full_prompt, ProblemSchema)

    async def create_solutions(self, problems: ProblemSchema) -> SolutionSchema:
        tools_list = ""
        if self.tool_manager and hasattr(self.tool_manager, "registry"):
            import json
            tools_info = self.tool_manager.registry.get_all_tools_info()
            tools_list = "\n\nAvailable Tools:\n" + json.dumps(tools_info, indent=2)

        system_prompt = (
            "You are the Thinker. Given the following defined problems, generate an algorithmic "
            "or architectural solution for each problem, including the tools required."
            f"{tools_list}"
        )
        problems_json = problems.json()
        full_prompt = f"{system_prompt}\n\nProblems:\n{problems_json}"
        
        return await self.llm.generate_structured(full_prompt, SolutionSchema, max_tokens=8192)

    async def generate_action_payload(self, solution: SolutionSchema) -> ActionSchema:
        tools_list = ""
        if self.tool_manager and hasattr(self.tool_manager, "registry"):
            import json
            tools_info = self.tool_manager.registry.get_all_tools_info()
            tools_list = "\n\nAvailable Tools:\n" + json.dumps(tools_info, indent=2)

        system_prompt = (
            "You are the Thinker. Given the following solutions, define the exact strict actions to take.\n"
            "Include the precise tools to call with arguments, and strict file I/O operations (file paths and contents) to enact the solution."
            f"{tools_list}\n\n"
            "IMPORTANT: When specifying tools_to_call, ONLY use the tool names provided in the Available Tools list. Do NOT invent new tool names."
        )
        solution_json = solution.json()
        full_prompt = f"{system_prompt}\n\nSolutions:\n{solution_json}"
        
        return await self.llm.generate_structured(full_prompt, ActionSchema, max_tokens=8192)

    async def generate_reflection(self, runtime_output: Any, context: str) -> ReflectionSchema:
        system_prompt = (
            "You are the Thinker reflecting on the output of the isolated runtime execution.\n"
            "Analyze the success, stdout, and stderr. Judge if the current task is complete.\n"
            "Provide detailed feedback and a rating."
        )
        full_prompt = f"{system_prompt}\n\nContext/Objective: {context}\n\nRuntime Output:\n{runtime_output}"
        
        return await self.llm.generate_structured(full_prompt, ReflectionSchema)
