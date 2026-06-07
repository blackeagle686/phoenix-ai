from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from ..schemas.brain import PlanSchema, ProblemSchema, SolutionSchema, ActionSchema, ReflectionSchema

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
    async def define_problems(self, task: Any) -> ProblemSchema:
        """
        Brain Step 2: Given a task, defines the specific problems to solve.
        """
        pass

    @abstractmethod
    async def create_solutions(self, problems: ProblemSchema) -> SolutionSchema:
        """
        Brain Step 3: Given defined problems, generates algorithmic/architectural solutions.
        """
        pass

    @abstractmethod
    async def generate_action_payload(self, solution: SolutionSchema) -> ActionSchema:
        """
        Brain Step 4: Given solutions, generates the explicit tools and file I/O operations.
        """
        pass

    @abstractmethod
    async def generate_reflection(self, runtime_output: Any, context: str) -> ReflectionSchema:
        """
        Brain Step 5: Analyzes runtime output to judge completion and provide feedback.
        """
        pass
