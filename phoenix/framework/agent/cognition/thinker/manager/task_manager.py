from phoenix.framework.agent.cognition.planner.schema import Task

class TaskManager:
    def __init__(self, ) -> Task:
        pass

    def push_task(self, task: TaskSchema) -> Task:
        pass

    def get_task(self, task_id: str) -> Task:
        pass

    def update_task(self, task: TaskSchema) -> None:
        pass

    def delete_task(self, task_id: str) -> None:
        pass

    def list_tasks(self) -> List[TaskSchema]:
        pass
    
    def clear_tasks(self) -> None:
        pass