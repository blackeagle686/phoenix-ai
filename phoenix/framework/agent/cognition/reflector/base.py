from abc import ABC, abstractmethod
from typing import Any, Optional
from phoenix.framework.agent.cognition.reflector.schema import ReflectorInputSchema, ReflectorOutputSchema

class BaseReflector(ABC):
    def __init__(self, llm: Any, profile: Optional[Any] = None):
        self.llm = llm
        self.profile = profile

    @abstractmethod
    async def reflect(self, input_data: ReflectorInputSchema) -> ReflectorOutputSchema:
        """
        Evaluates a task, problem, or solution based on the structured ReflectorInputSchema.
        Returns a ReflectorOutputSchema containing the rating, feedback, reasoning, and confidence.
        """
        pass
