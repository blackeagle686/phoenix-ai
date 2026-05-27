import json
import re
from typing import Dict, Any, AsyncGenerator, Optional
from .base import BasePlanner
from ..utils import parse_llm_json
from phoenix.services.cache import RedisCache
from phoenix.framework.agent.cognition.planner.schema import * 
from uuid import uuid4

async def get_redis_client():
    client = RedisCache()
    await client.init()
    return client 

task_cache = await get_redis_client()

class Planner(BasePlanner):
    """
    Generates actionable steps and selects tools based on the Thinker's objective.
    Enhanced with stateful task file support.
    """
    
    def __init__(self, llm, tools, task_store=task_cache, profile=None):
        super().__init__(llm, tools, task_store=task_store, profile=profile)
        self._cached_tool_info = None

    def _build_planner_prompt(
        self, 
        objective: str, 
        previous_results: str = "",
        existing_tasks: Dict[str, Any] = NoneNone
    ) -> str:
        if self._cached_tool_info is None:
            self._cached_tool_info = json.dumps(self.tools.get_all_tools_info(), indent=2)

        available_tools = self._cached_tool_info
        
        # Use BasePlanner helper to build context
        planning_context = self.build_planning_context(objective, previous_results, existing_tasks or {})

        system_prompt = f"""
        You are the 'Planner' module of an autonomous agent.
        Based on the provided context, formulate the next action using one of the available tools.
        
        Available Tools:
        {available_tools}
        
        Rules:
        1. Actions Over Talking: never claim completion unless verifiable action results show objective is complete.
        2. Verify Completion: use 'finish' only after at least one concrete tool action, except for pure conversational asks.
        3. Precision Editing: prefer file_read -> file_edit loops for existing files.
        4. Continue from existing tasks if available.
        
        You must respond with a JSON object strictly following this format:
        {{
            "actions": [
                {{"tool": "tool_name", "kwargs": {{"arg1": "value1"}}}},
                {{"tool": "tool_name", "kwargs": {{"arg1": "value1"}}}}
            ],
            "updated_tasks": {{ "task_id": {{ "status": "done", ... }} }}
        }}
        If you believe the task is complete, use "tool": "finish".
        """
        
        if self.profile:
            system_prompt += f"\n\n{self.profile.to_prompt_string()}"
            
        return f"{system_prompt}\n\n{planning_context}\n\nPlan (JSON only):"

    async def create_task(self, objective: str, user_) -> str:
        task_id = str(uuid4())
        task = Task(
            task_id=task_id,
            description=objective,
            status=TaskStatus.IN_PROGRESS
        )
        await self.task_store.set(key=f"task[{task_id}]", value=task.dict())
        return "task"+task_id


    async def stream_thinking(
        self, 
        objective: str, 
        task_file_id: Optional[str] = None,
        previous_results: str = ""
    ) -> AsyncGenerator[str, None]:
        
        existing_tasks = {}
        if task_file_id:
            existing_tasks = await self.load_task_file(task_file_id)

        thinking_prompt = f"""
        You are the Planner and must briefly explain your next-step reasoning before taking action.
        Objective: {objective}
        Existing Tasks: {existing_tasks}
        Previous results: {previous_results}

        Produce concise thought text.
        """

        stream_fn = getattr(self.llm, "generate_stream", None)
        if callable(stream_fn):
            try:
                stream = stream_fn(thinking_prompt, session_id=None, max_tokens=200)
                if hasattr(stream, "__aiter__"):
                    yielded = False
                    async for chunk in stream:
                        if chunk:
                            yielded = True
                            yield str(chunk)
                    if yielded:
                        return
            except Exception:
                pass

        text = await self.llm.generate(thinking_prompt, session_id=None, max_tokens=200)
        if text:
            for token in text.split():
                yield token + " "

    async def plan(
        self, 
        objective: str, 
        task_file_id: Optional[str] = None,
        previous_results: str = ""
    ) -> Dict[str, Any]:
        
        existing_tasks = {}
        if task_file_id:
            existing_tasks = await self.load_task_file(task_file_id)
            
        full_prompt = self._build_planner_prompt(objective, previous_results, existing_tasks)
        response = await self.llm.generate(full_prompt, session_id=None)
        
        data = parse_llm_json(response)
        if not data:
            return {"tool": "finish", "kwargs": {"reason": "Failed to parse planner output"}}
        
        # Auto-update task store if tasks were returned
        if task_file_id and "updated_tasks" in data:
            await self.update_task_file(task_file_id, data["updated_tasks"])
            
        return data
