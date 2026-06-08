from phoenix.framework.agent.cognition.planner.schema import Task

class PlanManager:
    def __init__(self, ):
        pass

    def push_task(self, task: TaskSchema) -> str:
        pass

    def get_task(self, task_id: str) -> TaskSchema:
        pass

    def update_task(self, task: TaskSchema, content) -> None:
        pass

    def delete_task(self, task_id: str) -> None:
        pass

    def list_tasks(self) -> List[TaskSchema]:
        pass
    
    def clear_tasks(self) -> None:
        pass