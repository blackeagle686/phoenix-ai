"""
    Task creator is responsible for creating tasks based on the objective.
    It uses the LLM to generate tasks from the objective.

"""

class Problem: 
    def __init__(self, description:str, )

class TaskCreator:
    def __init__(self, llm: Any, tools: Optional[Any] = None, ):
        self.llm = llm
        self.tools = tools
        self.memory = memory
        self.profile = profile

    async def create_task(self, objective: str, user_prompt: str) -> Task:
        """Creates a structured task based on objective and prompt."""
        pass