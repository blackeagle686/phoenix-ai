from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from ..schemas import FileTask, Task, ThinkerOutput

class BaseThinker(ABC):
    def __init__(self, llm, profile: Any = None):
        self.llm = llm
        self.profile = profile

    @abstractmethod
    async def analyze(self, prompt: str, memory: Any, session_id: str) -> str:
        """
        Full thinking pipeline (orchestrated)
        """
        pass

    @abstractmethod
    async def generate_main_objective(self, prompt: str) -> str:
        pass

    @abstractmethod
    async def generate_sub_objectives(self, main_objective: str) -> List[str]:
        pass

    @abstractmethod
    async def retrieve_context_memory(self, main_objective: str) -> List[str]:
        pass

    @abstractmethod
    async def summarize(self, prompt: str) -> str:
        pass

    def _build_output(
        self,
        main_objective: str,
        sub_objectives: List[str],
        context_memory: List[str],
        summary: str
    ) -> ThinkerOutput:

        return ThinkerOutput(
            main_objective=main_objective,
            sub_objectives=sub_objectives,
            context_memory=context_memory,
            summary_answer=summary,
            files={},
            tasks={}
        )

    def generate_file_task(self, file_path: str, operation: str) -> FileTask:
        return FileTask(file_path=file_path, operation=operation)

    def generate_task(
        self,
        task_id: str,
        description: str,
        dependencies: Optional[List[str]] = None,
        tools_required: Optional[List[str]] = None,
        priority: str = "medium"
    ) -> Task:

        return Task(
            task_id=task_id,
            description=description,
            dependencies=dependencies or [],
            tools_required=tools_required or [],
            priority=priority,
            status="pending"
        )
