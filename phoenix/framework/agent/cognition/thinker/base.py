from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from phoenix.framework.agent.cognition.schema import PlanSchema, TaskExecutionSchema, ReflectionSchema

class BaseThinker(ABC):
    def __init__(self, llm, profile: Any = None, tool_manager: Any = None):
        self.llm = llm
        self.profile = profile
        self.tool_manager = tool_manager

    @abstractmethod
    async def generate_plan(self, prompt: str, memory: Any, session_id: str) -> PlanSchema:
        """
        Brain Step 1: Takes the user prompt, utilizes tools to analyze the project,
        and generates a high-level plan with tasks.
        """
        pass

    @abstractmethod
    async def solve_task(self, task: Any, context: str) -> TaskExecutionSchema:
        """
        Brain Step 2: Solves the task by reasoning about problems, solutions, and generating actions (tool calls).
        """
        pass

    @abstractmethod
    async def generate_reflection(self, runtime_output: Any, context: str) -> ReflectionSchema:
        """
        Brain Step 5: Analyzes runtime output to judge completion and provide feedback.
        """
        pass
