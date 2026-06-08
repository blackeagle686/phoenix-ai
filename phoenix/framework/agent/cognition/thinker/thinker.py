import json
from typing import Any
from .base import BaseThinker
from phoenix.framework.agent.cognition.schema import PlanSchema, TaskExecutionSchema, ReflectionSchema
from phoenix.framework.agent.cognition.schema import (
    TaskType,
    TaskPriority,
    TaskStatus,
    ProblemComplexity,
    SolutionType,
    FileOperation
)


class Thinker(BaseThinker):
    """Central Brain. Only module that makes LLM generation calls"""

    def __init__(self, llm, profile=None, tool_manager=None):
        super().__init__(llm, profile=profile, tool_manager=tool_manager)

    def _get_tools_list(self) -> str:
        """Build available tools string for prompt injection"""
        if self.tool_manager and hasattr(self.tool_manager, "registry"):
            tools_info = self.tool_manager.registry.get_all_tools_info()
            return "\n\nAvailable Tools:\n" + json.dumps(tools_info, indent=2)
        return ""

    def _get_profile_string(self) -> str:
        """Get the agent profile prompt string"""
        if self.profile:
            return f"\n\n{self.profile.to_prompt_string()}"
        return ""

    async def generate_plan(self, prompt: str, memory: Any, session_id: str) -> PlanSchema:
        """Brain Step 1: Analyze user request and generate a structured plan with tasks"""
        context = await memory.get_full_context(session_id, query=prompt)
        task_types = [e.value for e in TaskType]
        task_priorities = [e.value for e in TaskPriority]
        task_statuses = [e.value for e in TaskStatus]

        system_prompt = (
            "You are the master Thinker and Planner. Analyze the user request and project context.\n"
            "Create a strict plan containing a main objective and a list of step by step tasks to accomplish it.\n"
            "IMPORTANT:\n"
            f" - For task_type, use one of: {task_types}.\n"
            f" - For priority, use one of: {task_priorities}.\n"
            f" - For status, use one of: {task_statuses}.\n"
            " - ALL file paths must be ABSOLUTE full paths. Never use relative paths."
        )
        system_prompt += self._get_profile_string()

        full_prompt = f"{system_prompt}\n\nContext:\n{context}\n\nUser Request: {prompt}"
        return await self.llm.generate_structured(full_prompt, PlanSchema, session_id=session_id, max_tokens=8192)

    async def solve_task(self, task: Any, context: str = "") -> TaskExecutionSchema:
        """Brain Step 2: Reason about the task and generate the strict tools to call"""
        tools_list = self._get_tools_list()
        task_desc = getattr(task, "description", str(task))
        task_id = getattr(task, "task_id", "unknown")

        system_prompt = (
            "You are the Thinker. Given the following task and its full context, analyze the problem, "
            "formulate an architectural/algorithmic approach, and define the exact explicit tool calls required to solve it.\n"
            f"{tools_list}\n\n"
            "IMPORTANT:\n"
            "1. ALWAYS use ABSOLUTE file paths for any file_path or directory. Never use relative paths.\n"
            "2. When specifying tools_to_call, ONLY use the tool names provided in the Available Tools list.\n"
            "3. Use native tool schemas like `file_write`, `file_edit`, `file_update_multi` to edit files.\n"
            "4. NEVER use shell commands (mkdir, touch, rm, cat, echo) for file operations. ALWAYS use file tools.\n"
            "5. If previous attempts failed, check the context for error details and adjust your approach."
        )

        full_prompt = f"{system_prompt}\n\nTask ID: {task_id}\nTask Description: {task_desc}\n\nContext:\n{context}"
        return await self.llm.generate_structured(full_prompt, TaskExecutionSchema, max_tokens=8192)

    async def generate_reflection(self, runtime_output: Any, context: str) -> ReflectionSchema:
        """Brain Step 5: Evaluate runtime execution results and judge task completion"""
        task_statuses = [e.value for e in TaskStatus]

        system_prompt = (
            "You are the Thinker reflecting on the output of the runtime execution.\n"
            "Analyze the success, stdout, stderr, and any IO or tool failures.\n"
            "Use the full context including previous attempts, failed operations, and error logs.\n"
            "Judge if the current task is complete. Provide detailed feedback and a rating.\n"
            "IMPORTANT:\n"
            f" - For status, use one of: {task_statuses}.\n"
            " - If IO operations failed, explain what went wrong and suggest corrections.\n"
            " - If tools were blocked, suggest the correct tool or io_operation to use."
        )

        full_prompt = f"{system_prompt}\n\nFull Context:\n{context}\n\nRuntime Output:\n{runtime_output}"
        return await self.llm.generate_structured(full_prompt, ReflectionSchema)
