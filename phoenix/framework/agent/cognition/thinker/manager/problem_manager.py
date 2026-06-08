from phoenix.framework.agent.cognition.planner.schema import Problem

class ProblemManager:
    def __init__(self, ) -> Problem:
        pass

    def push_problem(self, problem: Problem) -> Problem:
        pass

    def get_problem(self, problem_id: str) -> Problem:
        pass

    def update_problem(self, problem: Problem) -> None:
        pass

    def delete_problem(self, problem_id: str) -> None:
        pass

    def list_problems(self) -> List[Problem]:
        pass
    
    def clear_problems(self) -> None:
        pass