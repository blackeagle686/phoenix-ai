from typing import Any, Dict, List
from phoenix.framework.agent.cognition.schema import Task

class TaskManager:
    def __init__(self) -> None:
        self.tasks: Dict[str, Task] = {}

    def _check_task_id(self, task_id: str) -> None:
        if not isinstance(task_id, str):
            raise TypeError("task_id must be a string")
        if not task_id:
            raise ValueError("task_id cannot be empty")
        if task_id not in self.tasks:
            raise ValueError(f"Task with id {task_id} does not exist")

    def push_task(self, task: Task) -> Task:
        if not isinstance(task, Task):
            raise TypeError("task must be a Task instance")
        self.tasks[task.task_id] = task
        return task

    def get_task(self, task_id: str) -> Task:
        self._check_task_id(task_id)
        return self.tasks[task_id]

    def update_task(self, task: Task) -> None:
        if not isinstance(task, Task):
            raise TypeError("task must be a Task instance")
        self._check_task_id(task.task_id)
        self.tasks[task.task_id] = task

    def delete_task(self, task_id: str) -> None:
        self._check_task_id(task_id)
        del self.tasks[task_id]

    def list_tasks(self) -> List[Task]:
        return list(self.tasks.values())
    
    def clear_tasks(self) -> None:
        self.tasks.clear()