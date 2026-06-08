from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from phoenix.framework.agent.cognition.schema import (
    Prompt, PlannerInputSchema, PlannerOutputSchema,
    Task, TaskType, Problem, Solution
)
from phoenix.framework.agent.cognition.schema import BaseReflectorMeta

class BasePlanner(ABC): 
    def __init__(
        self, 
        thinker: Any, 
        tools: Optional[Any] = None, 
        memory: Optional[Any] = None,
        task_store: Optional[Any] = None, 
        profile: Optional[Any] = None
    ):
        self.thinker = thinker 
        self.tools = tools
        self.memory = memory 
        self.task_store = task_store
        self.profile = profile

    def build_planning_context(self, objective: str, previous_results: str, existing_tasks: Dict[str, Any]) -> str:
        """Helper to build standard planning context string."""
        context_parts = [f"OBJECTIVE: {objective}"]
        if existing_tasks:
            context_parts.append(f"EXISTING TASKS:\n{existing_tasks}")
        if previous_results:
            context_parts.append(f"PREVIOUS ACTION RESULTS:\n{previous_results}")
        return "\n\n".join(context_parts)

    async def load_task_file(self, task_file_id: str) -> Dict[str, Any]:
        """Loads tasks from Redis task store."""
        if self.task_store and hasattr(self.task_store, "get"):
            return await self.task_store.get(f"task_file[{task_file_id}]") or {}
        return {}

    async def update_task_file(self, task_file_id: str, tasks: Dict[str, Any]):
        """Saves tasks to Redis task store."""
        if self.task_store and hasattr(self.task_store, "set"):
            await self.task_store.set(f"task_file[{task_file_id}]", tasks)

    @abstractmethod
    async def plan(self, objective: str, task_file_id: Optional[str] = None, previous_results: str = "") -> Dict[str, Any]:
        """Core planning method to generate next actions."""
        pass

    @abstractmethod
    async def create_task(self, objective: str, user_prompt: str) -> Task:
        """Creates a structured task based on objective and prompt."""
        pass

    @abstractmethod
    def stream_thinking(self, objective: str, task_file_id: Optional[str] = None, previous_results: str = "") -> Any:
        """Streams planner reasoning/thoughts before acting."""
        pass
