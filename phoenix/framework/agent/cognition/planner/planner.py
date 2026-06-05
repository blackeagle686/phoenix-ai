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

class Planner(BasePlanner):
    """
    Generates actionable steps and selects tools based on the Thinker's objective.
    Enhanced with stateful task file support.
    """
    
    def __init__(self, llm, tools, task_store=None, profile=None):
        super().__init__(llm, tools, task_store=task_store, profile=profile)
        self._cached_tool_info = None

    async def _ensure_task_store(self):
        if self.task_store is None:
            self.task_store = RedisCache()
            await self.task_store.init()
        elif hasattr(self.task_store, "init") and getattr(self.task_store, "redis", None) is None:
            await self.task_store.init()

    def _build_planner_prompt(
        self, 
        objective: str, 
        previous_results: str = "",
        existing_tasks: Dict[str, Any] = None
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
        3. Precision Editing: prefer file_read -> file_edit loops for existing files. Use line-based chunks for edits.
        4. Continue from existing tasks if available.
        5. Directory Operations: DO NOT use file_read on directories. To create files in new directories, simply use file_write with the desired path. To explore, use execute_command (e.g. 'ls').
        You must respond with a JSON object strictly following this format:
        {{
            "actions": [
                {{"tool": "file_read", "kwargs": {{"file_path": "path/to/file.py", "from_line": 1, "to_line": 100}}}},
                {{"tool": "file_write", "kwargs": {{"file_path": "path/to/output.txt", "content": "hello world"}}}}
            ],
            "updated_tasks": {{
                "task_id_1": {{
                    "task_id": "task_id_1",
                    "task_summary": "Summary of the task",
                    "description": "Description of what needs to be done",
                    "dependencies": [],
                    "tools_required": ["file_read", "file_edit"],
                    "priority": "medium",
                    "status": "done",
                    "output": "Optional result output of this task",
                    "file_tasks": [
                        {{"file_path": "path/to/file.py", "operation": "edit"}}
                    ]
                }}
            }}
        }}
        If you believe the task is complete, use "tool": "finish".
        """
        
        if self.profile:
            system_prompt += f"\n\n{self.profile.to_prompt_string()}"
            
        return f"{system_prompt}\n\n{planning_context}\n\nPlan (JSON only):"

    async def get_dependecies(self) -> List[str]:
        await self._ensure_task_store()
        tasks = await self.task_store.get_all("task[*]")
        
        tasks_summary = []
        for task in tasks.values():
            if isinstance(task, dict):
                t_id = task.get("task_id")
                t_summary = task.get("task_summary") or task.get("summary")
                t_status = task.get("status")
            else:
                t_id = getattr(task, "task_id", None)
                t_summary = getattr(task, "task_summary", None)
                t_status = getattr(task, "status", None)
                
            if t_id and t_status != TaskStatus.DONE:
                tasks_summary.append({
                    "task_id": t_id,
                    "summary": t_summary
                })

        prompt = f"""
You are an expert dependency analyzer. Given a list of active task summaries, analyze them and determine which task IDs are dependencies that must be resolved first.
Tasks Summary: {json.dumps(tasks_summary)}

Respond with a JSON list only containing the dependency task IDs. Format:
[
  "task_id_1",
  "task_id_2"
]
"""
        response = await self.llm.generate(prompt)
        data = parse_llm_json(response)
        
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and "dependencies" in data:
            return data["dependencies"]
            
        # Fallback to programmatic lookup of dependencies
        deps = set()
        for task in tasks.values():
            if isinstance(task, dict):
                deps.update(task.get("dependencies", []))
            elif hasattr(task, "dependencies"):
                deps.update(getattr(task, "dependencies", []))
        return list(deps)

    async def create_task(self, objective: str, user_prompt: str) -> Task:
        from phoenix.framework.agent.cognition.planner.task_creator import TaskCreator
        await self._ensure_task_store()
        
        task_creator = TaskCreator(llm=self.llm, tools=self.tools, cache=self.task_store)
        task = await task_creator.create_task(objective, user_prompt)
        
        await self.task_store.set(key=f"task[{task.task_id}]", value=task.model_dump())
        return task


    async def stream_thinking(
        self, 
        objective: str, 
        task_file_id: Optional[str] = None,
        previous_results: str = ""
    ) -> AsyncGenerator[str, None]:
        await self._ensure_task_store()
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

        import inspect
        stream_fn = getattr(self.llm, "generate_stream", None)
        if callable(stream_fn):
            try:
                if inspect.iscoroutinefunction(stream_fn):
                    stream = await stream_fn(thinking_prompt, session_id=None, max_tokens=200)
                else:
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
        await self._ensure_task_store()
        existing_tasks = {}
        if task_file_id:
            existing_tasks = await self.load_task_file(task_file_id)
            
        full_prompt = self._build_planner_prompt(objective, previous_results, existing_tasks)
        response = await self.llm.generate(full_prompt, session_id=None)
        
        data = parse_llm_json(response)
        if not data:
            return {"tool": "finish", "kwargs": {"reason": "Failed to parse planner output"}}
        
        # Validate and coerce updated_tasks to match the new Task Pydantic schema
        if "updated_tasks" in data and isinstance(data["updated_tasks"], dict):
            validated_tasks = {}
            for t_id, t_data in data["updated_tasks"].items():
                if isinstance(t_data, dict):
                    t_data.setdefault("task_id", t_id)
                    try:
                        validated_task_obj = Task(**t_data)
                        validated_tasks[t_id] = validated_task_obj.dict()
                    except Exception:
                        validated_tasks[t_id] = t_data
                else:
                    validated_tasks[t_id] = t_data
            data["updated_tasks"] = validated_tasks

        # Auto-update task store if tasks were returned
        if task_file_id and "updated_tasks" in data:
            await self.update_task_file(task_file_id, data["updated_tasks"])
            
        # Standardize and map legacy arguments for actions
        if "actions" in data and isinstance(data["actions"], list):
            for action in data["actions"]:
                tool_name = action.get("tool")
                kwargs = action.get("kwargs", {})
                if tool_name == "file_search":
                    if "path" in kwargs and "file_path" not in kwargs:
                        kwargs["file_path"] = kwargs.pop("path")
                    if "pattern" in kwargs and "search_query" not in kwargs:
                        kwargs["search_query"] = kwargs.pop("pattern")
            
        return data

if __name__ == "__main__":
    import asyncio
    from phoenix.services.llm.openai import OpenAILLM

    llm = OpenAILLM(
        api_key="ak_2yp3Xw1Ny7ky2pF7er9x93ZO9jj6G",
        model="LongCat-2.0-Preview",
        base_url="https://api.longcat.chat/openai"
    )

    class MockTools:
        def __init__(self):
            self.tools = {
                "file_write": "Write code or text to a specific file path",
                "file_read": "Read content from a specific file path",
                "execute_terminal": "Run terminal commands",
                "folder_create": "Create a new folder"
            }
        def get_all_tools_info(self):
            return [{"name": k, "description": v} for k, v in self.tools.items()]

    async def run_test():
        await llm.init()
        
        mock_tools = MockTools()
        
        # Initialize an empty cache for task store simulation
        from phoenix.services.cache import RedisCache
        task_store = RedisCache()
        await task_store.init()
        
        planner = Planner(llm=llm, tools=mock_tools, task_store=task_store)
        
        objective = "build a secure rust actix-web API for a blogging engine"
        prompt = "create the initial architecture tasks"
        
        print("Starting Planner create_task test using TaskCreator...")
        task = await planner.create_task(objective, prompt)
        
        print("\n" + "="*60)
        print("PLANNER GENERATED TASK (Readable Format)")
        print("="*60)
        print(task.model_dump_json(indent=5))
        print("="*60 + "\n")

    asyncio.run(run_test())