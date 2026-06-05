"""
    Task creator is responsible for creating tasks based on the objective.
    It uses the LLM to generate tasks from the objective.

"""



class TaskCreator:
    def __init__(self, llm: Any, tools: Optional[Any] = None, cache: Optional[Any] = None ):
        self.llm = llm
        self.tools = tools
        self.cache = cache
        if not cache: 
            self.memory = {}

    async def create_problem(self, objective: str) -> Problem:
        pass

    async def create_solution(slef, problem):
        pass
    
    async def create_task(self, objective: str, user_prompt: str) -> Task:
        """Creates a structured task based on objective and prompt."""
        pass