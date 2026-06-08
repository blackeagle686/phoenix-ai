from typing import Dict, List, Union
from uuid import UUID
from phoenix.framework.agent.cognition.planner.schema import Problem

class ProblemManager:
    def __init__(self) -> None:
        self.problems: Dict[str, Problem] = {}

    def _check_problem_id(self, problem_id: Union[str, UUID]) -> str:
        if isinstance(problem_id, UUID):
            problem_id = str(problem_id)
        if not isinstance(problem_id, str):
            raise TypeError("problem_id must be a string or UUID")
        if not problem_id:
            raise ValueError("problem_id cannot be empty")
        if problem_id not in self.problems:
            raise ValueError(f"Problem with id {problem_id} does not exist")
        return problem_id

    def push_problem(self, problem: Problem) -> Problem:
        if not isinstance(problem, Problem):
            raise TypeError("problem must be a Problem instance")
        self.problems[str(problem.id)] = problem
        return problem

    def get_problem(self, problem_id: Union[str, UUID]) -> Problem:
        problem_id_str = self._check_problem_id(problem_id)
        return self.problems[problem_id_str]

    def update_problem(self, problem: Problem) -> None:
        if not isinstance(problem, Problem):
            raise TypeError("problem must be a Problem instance")
        problem_id_str = self._check_problem_id(problem.id)
        self.problems[problem_id_str] = problem

    def delete_problem(self, problem_id: Union[str, UUID]) -> None:
        problem_id_str = self._check_problem_id(problem_id)
        del self.problems[problem_id_str]

    def list_problems(self) -> List[Problem]:
        return list(self.problems.values())
    
    def clear_problems(self) -> None:
        self.problems.clear()