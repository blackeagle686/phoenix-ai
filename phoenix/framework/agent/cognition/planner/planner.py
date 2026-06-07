import json
from typing import Dict, Any, Optional
from .base import BasePlanner
from phoenix.services.cache import RedisCache
from phoenix.framework.agent.cognition.schemas.brain import PlanSchema, ProblemSchema

class Planner(BasePlanner):
    """
    Manages the overall Plan. Delegates the generation of the plan and problems to the Thinker.
    """
    
    def __init__(self, thinker, tools=None, task_store=None, profile=None):
        super().__init__(thinker, tools, task_store=task_store, profile=profile)

    async def _ensure_task_store(self):
        if self.task_store is None:
            self.task_store = RedisCache()
            await self.task_store.init()
        elif hasattr(self.task_store, "init") and getattr(self.task_store, "redis", None) is None:
            await self.task_store.init()

    async def load_plan(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Loads the current plan from Redis."""
        await self._ensure_task_store()
        if self.task_store and hasattr(self.task_store, "get"):
            return await self.task_store.get(f"plan[{session_id}]")
        return None

    async def save_plan(self, session_id: str, plan: PlanSchema):
        """Saves the plan to Redis."""
        await self._ensure_task_store()
        if self.task_store and hasattr(self.task_store, "set"):
            await self.task_store.set(f"plan[{session_id}]", plan.dict())

    async def generate_initial_plan(self, prompt: str, memory: Any, session_id: str) -> PlanSchema:
        """
        Brain Step 1: Generates the full plan based on the user prompt.
        """
        plan = await self.thinker.generate_plan(prompt, memory, session_id)
        await self.save_plan(session_id, plan)
        return plan

    async def define_task_problems(self, task: Any) -> ProblemSchema:
        """
        Brain Step 2: For a specific task, defines the explicit problems to solve.
        """
        problems = await self.thinker.define_problems(task)
        return problems

    # Keeping legacy method signatures for backward compatibility, 
    # but pointing them to the new schema-based flow if needed.
    async def create_task(self, objective: str, user_prompt: str) -> Any:
        pass

    async def plan(self, objective: str, task_file_id: Optional[str] = None, previous_results: str = "") -> Dict[str, Any]:
        return {}

    async def stream_thinking(self, objective: str, task_file_id: Optional[str] = None, previous_results: str = ""):
        yield "Planner is orchestrating the Thinker to define problems..."