from abc import ABC, abstractmethod
from typing import Any, Optional
from enum import Enum
from phoenix.framework.agent.cognition.schema import ActorInputSchema, ActorOutputSchema, Task
from phoenix.framework.agent.runtime.base import BaseRuntime


class Execution_type(str, Enum):
    COMPILE = "compile"
    RUN = "run"


class BaseActor(ABC):
    def __init__(
        self,
        tool_manager: Any,
        thinker: Optional[Any] = None,
        runtime: Optional[BaseRuntime] = None,
        reflector: Optional[Any] = None,
    ):
        self.tool_manager = tool_manager
        self.thinker = thinker
        self.reflector = reflector
        self.runtime = runtime

    @abstractmethod
    async def execute(self, task_input: Any, task_context: Optional[Task] = None) -> Any:
        """
        Executes a task or action batch using the tool manager.
        """
        pass

    @abstractmethod
    async def generate_and_execute(self, task: Any, previous_results: str = "", context: str = "") -> ActorOutputSchema:
        """
        Dynamically generates the required tool calls via Thinker and executes them.
        """
        pass
