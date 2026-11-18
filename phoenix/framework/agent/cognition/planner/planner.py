from __future__ import annotations
import inspect
import json
from typing import Dict, Any, Optional, TYPE_CHECKING
from .base import BasePlanner
from phoenix.framework.agent.cognition.thinker import Thinker
from phoenix.framework.agent.cognition.utils import parse_llm_json

if TYPE_CHECKING:
    from phoenix.framework.agent.cognition.schema import PlanSchema, ProblemSchema


class Planner(BasePlanner):
    """
    Manages the overall Plan and step-level action planning.
    Delegates structured PlanSchema generation to the Thinker, with resilient caching.
    """

    def __init__(
        self,
        thinker: Any = None,
        tools: Any = None,
        task_store: Any = None,
        profile: Any = None,
        llm: Any = None,
        memory: Any = None,
    ):
        raw_llm = llm
        if thinker is not None and not hasattr(thinker, "generate_plan") and hasattr(thinker, "generate"):
            raw_llm = thinker
            actual_thinker = Thinker(raw_llm, profile=profile)
        elif thinker is None and raw_llm is not None:
            actual_thinker = Thinker(raw_llm, profile=profile)
        else:
            actual_thinker = thinker
            if raw_llm is None and actual_thinker is not None:
                raw_llm = getattr(actual_thinker, "llm", None)

        super().__init__(actual_thinker, tools=tools, memory=memory, task_store=task_store, profile=profile)
        if actual_thinker is not None and tools is not None:
            if getattr(actual_thinker, "tool_manager", None) is None and getattr(actual_thinker, "tools", None) is None:
                actual_thinker.tools = tools
        self.llm = raw_llm
        self._local_plan_cache: Dict[str, Any] = {}

    async def _ensure_task_store(self):
        if self.task_store is not None:
            if hasattr(self.task_store, "init") and getattr(self.task_store, "redis", None) is None:
                try:
                    await self.task_store.init()
                except Exception:
                    pass

    async def load_plan(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Loads the current plan from task_store or in-memory fallback."""
        await self._ensure_task_store()
        if self.task_store and hasattr(self.task_store, "get"):
            try:
                cached = await self.task_store.get(f"plan[{session_id}]")
                if cached is not None:
                    return cached
            except Exception:
                pass
        return self._local_plan_cache.get(f"plan[{session_id}]")

    async def save_plan(self, session_id: str, plan: PlanSchema):
        """Saves the plan to task_store and in-memory fallback."""
        plan_data = plan.model_dump() if hasattr(plan, "model_dump") else (plan.dict() if hasattr(plan, "dict") else plan)
        self._local_plan_cache[f"plan[{session_id}]"] = plan_data
        await self._ensure_task_store()
        if self.task_store and hasattr(self.task_store, "set"):
            try:
                await self.task_store.set(f"plan[{session_id}]", plan_data)
            except Exception:
                pass

    async def generate_initial_plan(
        self,
        prompt: str,
        memory: Any = None,
        session_id: str = "",
        workspace_analysis: Optional[Dict[str, Any]] = None,
    ) -> PlanSchema:
        """
        Brain Step 1: Generates the full plan based on the user prompt and optional workspace analysis.
        """
        sig = inspect.signature(self.thinker.generate_plan)
        if "workspace_analysis" in sig.parameters:
            plan = await self.thinker.generate_plan(
                prompt,
                memory,
                session_id,
                workspace_analysis=workspace_analysis,
            )
        else:
            plan = await self.thinker.generate_plan(prompt, memory, session_id)
        await self.save_plan(session_id, plan)
        return plan

    async def create_task(self, objective: str, user_prompt: str) -> Any:
        from .task_creator import TaskCreator
        creator = TaskCreator(thinker=self.thinker, tools=self.tools, llm=self.llm)
        return await creator.create_task(objective, user_prompt)

    async def plan(
        self,
        objective: str,
        previous_results: str = "",
        task_file_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generates a dictionary of next actions for iterative planning loops."""
        tools_info = []
        if self.tools and hasattr(self.tools, "get_all_tools_info"):
            tools_info = self.tools.get_all_tools_info()

        existing_tasks = await self.load_task_file(task_file_id) if task_file_id else {}
        context = self.build_planning_context(objective, previous_results, existing_tasks)
        prompt = (
            f"You are the Planner.\n{context}\n\n"
            f"Available Tools: {json.dumps(tools_info)}\n\n"
            "Respond with JSON containing an 'actions' list of {'tool': '...', 'kwargs': {...}}."
        )
        if self.llm and hasattr(self.llm, "generate"):
            raw = await self.llm.generate(prompt)
            parsed = parse_llm_json(raw)
            if parsed:
                return parsed
        return {"actions": []}

    async def stream_thinking(
        self,
        objective: str,
        previous_results: str = "",
        task_file_id: Optional[str] = None,
    ):
        """Streams planner reasoning thoughts before acting."""
        prompt = (
            f"Briefly explain your step-by-step thinking to achieve the objective.\n"
            f"Objective: {objective}\nPrevious Results: {previous_results}"
        )
        if self.llm:
            gen_stream = getattr(self.llm, "generate_stream", None)
            if gen_stream is not None and callable(gen_stream):
                stream_iter = gen_stream(prompt)
                if hasattr(stream_iter, "__aiter__"):
                    async for chunk in stream_iter:
                        yield chunk
                    return
            if hasattr(self.llm, "generate"):
                thought = await self.llm.generate(prompt)
                if thought:
                    yield str(thought)
                    return
        yield "Planner is orchestrating the Thinker to define tasks..."
