from __future__ import annotations
from typing import Any, Optional, TYPE_CHECKING
from .base import BaseReflector
from phoenix.framework.agent.cognition.schema import ReflectorInputSchema
from phoenix.framework.agent.cognition.thinker import Thinker
from phoenix.framework.agent.cognition.utils import parse_llm_json

if TYPE_CHECKING:
    from phoenix.framework.agent.cognition.schema import ReflectionSchema


class Reflector(BaseReflector):
    """
    Evaluates the quality of tasks, problems, and solutions by delegating
    to the centralized Thinker module (with fallback support for direct dict evaluation).
    """

    def __init__(self, thinker: Any = None, profile: Optional[Any] = None, llm: Optional[Any] = None):
        raw_llm = llm
        if thinker is not None and not hasattr(thinker, "generate_reflection") and hasattr(thinker, "generate"):
            raw_llm = thinker
            actual_thinker = Thinker(raw_llm, profile=profile)
        elif thinker is None and raw_llm is not None:
            actual_thinker = Thinker(raw_llm, profile=profile)
        else:
            actual_thinker = thinker
            if raw_llm is None and actual_thinker is not None:
                raw_llm = getattr(actual_thinker, "llm", None)

        super().__init__(actual_thinker, profile=profile)
        self.llm = raw_llm

    async def reflect(self, input_data: Any, action: Any = None, result: Any = None) -> Any:
        """
        Judges the execution result and decides on completion.
        Supports both `ReflectorInputSchema` (returning `ReflectionSchema`) and
        legacy `(objective, action, result)` arguments (returning `dict`).
        """
        if isinstance(input_data, ReflectorInputSchema):
            if not self.thinker:
                raise ValueError("Reflector requires a Thinker instance.")
            context = input_data.context or "No context provided"
            return await self.thinker.generate_reflection(
                runtime_output=input_data.target_content,
                context=context,
            )

        # Legacy 3-argument signature: reflect(objective, action, result)
        if self.llm and hasattr(self.llm, "generate"):
            prompt = (
                f"Evaluate whether the objective is complete based on the action and result.\n"
                f"Objective: {input_data}\nAction: {action}\nResult: {result}\n"
                f"Respond with JSON: {{\"is_complete\": true/false, \"reflection\": \"...\"}}"
            )
            raw = await self.llm.generate(prompt)
            parsed = parse_llm_json(raw)
            if parsed:
                return parsed
            return {"is_complete": False, "reflection": str(raw)}

        if self.thinker:
            ref = await self.thinker.generate_reflection(
                runtime_output={"action": action, "result": result},
                context=str(input_data),
            )
            return {
                "is_complete": ref.is_task_complete,
                "reflection": ref.feedback,
                "rating": ref.rating,
                "status": ref.status,
            }
        raise ValueError("Reflector requires a Thinker or LLM instance.")
