from phoenix.framework.agent.cognition.planner.schema import Task

class TaskManager:
    def __init__(self, ) -> Task:
        pass

    def _check_task_id(self, task_id: str) -> None:
        if not isinstance(task_id, str):
            raise TypeError("task_id must be a string")
        if not task_id:
            raise ValueError("task_id cannot be empty")

    def push_task(self, task: Task) -> Task:
        pass

    def get_task(self, task_id: str) -> Task:
        pass

    def update_task(self, task: Task) -> None:
        pass

    def delete_task(self, task_id: str) -> None:
        pass

    def list_tasks(self) -> List[TaskSchema]:
        pass
    
    def clear_tasks(self) -> None:
        pass