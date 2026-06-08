from __future__ import annotations
from typing import Any, TYPE_CHECKING
from .base import BaseReflector
from phoenix.framework.agent.cognition.schema import ReflectorInputSchema

if TYPE_CHECKING:
    from phoenix.framework.agent.cognition.schema import ReflectionSchema

class Reflector(BaseReflector):
    """
    Evaluates the quality of tasks, problems, and solutions by delegating
    to the centralized Thinker module.
    """

    def __init__(self, thinker, profile=None):
        super().__init__(thinker, profile=profile)

    async def reflect(self, input_data: ReflectorInputSchema) -> ReflectionSchema:
        """
        Brain Step 5: Judges the execution result and decides on completion.
        Delegates the heavy LLM call to the Thinker.
        """
        if not self.thinker:
            raise ValueError("Reflector requires a Thinker instance.")
            
        context = input_data.context or "No context provided"
        # We pass the input_data.target_content directly as runtime_output
        return await self.thinker.generate_reflection(
            runtime_output=input_data.target_content,
            context=context
        )
