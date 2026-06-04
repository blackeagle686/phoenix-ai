from abc import ABC, abstractmethod
from typing import Any, Optional
from phoenix.framework.agent.cognition.actor.schema import ActorInputSchema, ActorOutputSchema
from phoenix.framework.agent.cognition.planner.schema import Task

class BaseActor(ABC):
    def __init__(self, tool_manager: Any, reflector: Optional[Any] = None):
        self.tool_manager = tool_manager
        self.reflector = reflector

    @abstractmethod
    async def execute(self, task_input: ActorInputSchema, task_context: Optional[Task] = None) -> ActorOutputSchema:
        """
        Executes a task using the tool manager based on strict input schemas.
        Optionally takes a full Task context to generate reflections via the Reflector.
        """
        pass
