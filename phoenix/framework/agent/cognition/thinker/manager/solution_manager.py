from typing import Dict, List, Union
from uuid import UUID
from phoenix.framework.agent.cognition.planner.schema import Solution

class SolutionManager:
    def __init__(self) -> None:
        self.solutions: Dict[str, Solution] = {}

    def _check_solution_id(self, solution_id: Union[str, UUID]) -> str:
        if isinstance(solution_id, UUID):
            solution_id = str(solution_id)
        if not isinstance(solution_id, str):
            raise TypeError("solution_id must be a string or UUID")
        if not solution_id:
            raise ValueError("solution_id cannot be empty")
        if solution_id not in self.solutions:
            raise ValueError(f"Solution with id {solution_id} does not exist")
        return solution_id

    def push_solution(self, solution: Solution) -> Solution:
        if not isinstance(solution, Solution):
            raise TypeError("solution must be a Solution instance")
        self.solutions[str(solution.id)] = solution
        return solution

    def get_solution(self, solution_id: Union[str, UUID]) -> Solution:
        solution_id_str = self._check_solution_id(solution_id)
        return self.solutions[solution_id_str]

    def update_solution(self, solution: Solution) -> None:
        if not isinstance(solution, Solution):
            raise TypeError("solution must be a Solution instance")
        solution_id_str = self._check_solution_id(solution.id)
        self.solutions[solution_id_str] = solution

    def delete_solution(self, solution_id: Union[str, UUID]) -> None:
        solution_id_str = self._check_solution_id(solution_id)
        del self.solutions[solution_id_str]

    def list_solutions(self) -> List[Solution]:
        return list(self.solutions.values())
    
    def clear_solutions(self) -> None:
        self.solutions.clear()
