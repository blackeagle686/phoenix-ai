from abc import ABC, abstractmethod
from typing import Dict, Any, AsyncGenerator, List, Optional
from sch

class BasePlanner(ABC): 
    def __init__(self, llm, memory):
        self.llm = llm 
        self.memory = memory 

    def user_input(self, prompt:Prompt): 
        pass

    def actor_input(self): 
        pass

    def reflector_input(self):
        pass 

    def user_output(self):
        pass

    def actor_output(self):
        pass
    
    def reflector_output(self):
        pass
    
    def analysis_input(self): 
        pass 

    def analysis_output(self): 
        pass 

    def __task_builder(self, task_type): 
        pass 

    def __problem_builder(self): 
        pass 

    def __solution_builder(self): 
        pass 

    

class BasePlanner(ABC):
    def __init__(self, llm, tools, task_store=None, profile: Any = None):
        self.llm = llm
        self.tools = tools
        self.task_store = task_store  # 👈 NEW (memory/task system)
        self.profile = profile

    # =========================
    # Core Planning
    # =========================

    @abstractmethod
    async def plan(
        self,
        objective: str,
        task_file_id: Optional[str] = None,
        previous_results: str = ""
    ) -> Dict[str, Any]:
        """
        Generate or update plan based on:
        - objective
        - existing task file (stateful planning)
        """
        pass

    # =========================
    # Streaming Thinking
    # =========================

    @abstractmethod
    async def stream_thinking(
        self,
        objective: str,
        task_file_id: Optional[str] = None,
        previous_results: str = ""
    ) -> AsyncGenerator[str, None]:
        pass

    # =========================
    # Task File Integration Layer
    # =========================

    async def load_task_file(self, task_file_id: str) -> Dict[str, Any]:
        """
        Load existing task state from memory/store
        """
        if not self.task_store:
            return {}

        return await self.task_store.get(task_file_id)

    async def update_task_file(
        self,
        task_file_id: str,
        tasks: Dict[str, Any]
    ) -> None:
        """
        Persist updated task state
        """
        if not self.task_store:
            return

        await self.task_store.update(task_file_id, tasks)

    # =========================
    # Task State Helpers
    # =========================

    def filter_pending_tasks(self, tasks: Dict[str, Any]) -> Dict[str, Any]:
        return {
            k: v for k, v in tasks.items()
            if v.get("status") == "pending"
        }

    def get_in_progress_tasks(self, tasks: Dict[str, Any]) -> Dict[str, Any]:
        return {
            k: v for k, v in tasks.items()
            if v.get("status") == "in_progress"
        }

    def mark_task_done(self, tasks: Dict[str, Any], task_id: str):
        if task_id in tasks:
            tasks[task_id]["status"] = "done"

    # =========================
    # Context Builder
    # =========================

    def build_planning_context(
        self,
        objective: str,
        previous_results: str,
        existing_tasks: Dict[str, Any]
    ) -> str:

        return f"""
OBJECTIVE:
{objective}

PREVIOUS RESULTS:
{previous_results}

EXISTING TASK STATE:
{existing_tasks}

INSTRUCTIONS:
- Continue from existing tasks if available
- Do NOT recreate completed tasks
- Update only necessary tasks
- Add new tasks only if required
"""

