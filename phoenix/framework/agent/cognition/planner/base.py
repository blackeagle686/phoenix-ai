from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from phoenix.framework.agent.cognition.planner.schema import (
    Prompt, PlannerInputSchema, PlannerOutputSchema,
    Task, TaskType, Problem, Solution
)
from phoenix.framework.agent.cognition.actor.schema import (
    BaseTaskInputSchema, BaseTaskOutputSchema,
    BaseFileTaskInputSchema, BaseFileTaskOutputSchema
)
from phoenix.framework.agent.cognition.reflector.schema import BaseReflectorMeta

class BasePlanner(ABC): 
    def __init__(
        self, 
        llm: Any, 
        memory: Any, 
        tools: Optional[Any] = None, 
        task_store: Optional[Any] = None, 
        profile: Optional[Any] = None
    ):
        self.llm = llm 
        self.memory = memory 
        self.tools = tools
        self.task_store = task_store
        self.profile = profile

    # ==========================================
    # User Flow
    # ==========================================
    @abstractmethod
    def user_input(self, prompt: Prompt) -> PlannerInputSchema: 
        """Transforms a raw user Prompt into structured Planner input context."""
        pass

    @abstractmethod
    def user_output(self, output: PlannerOutputSchema) -> str:
        """Transforms the Planner's final state into a user-facing string response."""
        pass

    # ==========================================
    # Actor Flow
    # ==========================================
    @abstractmethod
    def actor_input(self, task: Task) -> BaseTaskInputSchema: 
        """Prepares a strict execution schema to send to an Actor."""
        pass

    @abstractmethod
    def actor_output(self, result: BaseTaskOutputSchema) -> Task:
        """Processes the Actor's result and updates the internal Task state."""
        pass
    
    # ==========================================
    # Reflector Flow
    # ==========================================
    @abstractmethod
    def reflector_input(self, item: Union[Task, Problem, Solution]) -> Dict[str, Any]:
        """Prepares data to send to the Reflector to evaluate a task, problem, or solution."""
        pass 

    @abstractmethod
    def reflector_output(self, evaluation: BaseReflectorMeta) -> Union[Task, Problem, Solution]:
        """Integrates Reflector feedback and ratings back into the original item."""
        pass
    
    # ==========================================
    # Analysis Flow
    # ==========================================
    @abstractmethod
    def analysis_input(self, task: BaseFileTaskInputSchema) -> BaseFileTaskInputSchema:
        """Prepares strict schema to read and understand files."""
        pass 

    @abstractmethod
    def analysis_output(self, result: BaseFileTaskOutputSchema) -> BaseFileTaskOutputSchema:
        """Processes the structured file analysis result."""
        pass 

    # ==========================================
    # Builders
    # ==========================================
    @abstractmethod
    def _task_builder(self, task_type: TaskType, **kwargs) -> Task: 
        pass 

    @abstractmethod
    def _problem_builder(self, **kwargs) -> Problem: 
        pass 

    @abstractmethod
    def _solution_builder(self, **kwargs) -> Solution: 
        pass 
