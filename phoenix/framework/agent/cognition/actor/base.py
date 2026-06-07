from abc import ABC, abstractmethod
from typing import Any, Optional
from phoenix.framework.agent.cognition.actor.schema import ActorInputSchema, ActorOutputSchema
from phoenix.framework.agent.cognition.planner.schema import Task
from enum import Enum
from phoenix.framework.agent.runtime.base import BaseRuntime
    COMPILE = "compile"
    RUN = "run"
        

class BaseActor(ABC):
    def __init__(self, tool_manager: Any, llm: Optional[Any] = None, reflector: Optional[Any] = None, runtime: Optional[BaseRuntime] = None):
        self.tool_manager = tool_manager
        self.llm = llm
        self.reflector = reflector
        self.runtime = runtime

    @abstractmethod
    async def execute(self, task_input: ActorInputSchema, task_context: Optional[Task] = None) -> ActorOutputSchema:
        """
        Executes a task using the tool manager based on strict input schemas.
        Optionally takes a full Task context to generate reflections via the Reflector.
        """
        pass

    @abstractmethod
    async def generate_and_execute(self, task: Task, previous_results: str = "") -> ActorOutputSchema:
        """
        Dynamically generates the required code and tool payload based on the Task's high-level strategy,
        then executes the tool immediately.
        """
        pass
