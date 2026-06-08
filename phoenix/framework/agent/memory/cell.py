from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import time

from phoenix.framework.agent.cognition.utils.id import generate_unique_id


@dataclass
class MemoryCell:
    """Single unified node holding all related context for one task lifecycle"""

    cell_id: str = field(default_factory=generate_unique_id)
    task_id: str = ""
    task_description: str = ""
    task_type: str = "other"
    task_status: str = "pending"
    task_priority: str = "medium"

    objective: str = ""
    dependencies: List[str] = field(default_factory=list)

    problems: List[Dict[str, Any]] = field(default_factory=list)
    solutions: List[Dict[str, Any]] = field(default_factory=list)
    actions: List[Dict[str, Any]] = field(default_factory=list)
    reflections: List[Dict[str, Any]] = field(default_factory=list)
    runtime_results: List[Dict[str, Any]] = field(default_factory=list)

    io_history: List[Dict[str, Any]] = field(default_factory=list)
    tool_history: List[Dict[str, Any]] = field(default_factory=list)
    error_log: List[str] = field(default_factory=list)

    attempts: int = 0
    max_attempts: int = 5
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def touch(self):
        """Update the timestamp"""
        self.updated_at = time.time()

    def add_problem(self, problem: Dict[str, Any]):
        """Push a problem definition into this cell"""
        self.problems.append(problem)
        self.touch()

    def add_solution(self, solution: Dict[str, Any]):
        """Push a solution definition into this cell"""
        self.solutions.append(solution)
        self.touch()

    def add_action(self, action: Dict[str, Any]):
        """Push an action record into this cell"""
        self.actions.append(action)
        self.touch()

    def add_reflection(self, reflection: Dict[str, Any]):
        """Push a reflection record into this cell"""
        self.reflections.append(reflection)
        self.touch()

    def add_runtime_result(self, result: Dict[str, Any]):
        """Push a runtime execution result into this cell"""
        self.runtime_results.append(result)
        self.touch()

    def add_io_record(self, operation: str, file_path: str, success: bool, error: Optional[str] = None):
        """Record a file IO operation that was executed"""
        self.io_history.append({
            "operation": operation,
            "file_path": file_path,
            "success": success,
            "error": error,
            "timestamp": time.time()
        })
        self.touch()

    def add_tool_record(self, tool_name: str, arguments: Dict[str, Any], success: bool, output: Any = None, error: Optional[str] = None):
        """Record a tool execution"""
        self.tool_history.append({
            "tool_name": tool_name,
            "arguments": arguments,
            "success": success,
            "output": str(output)[:2000] if output else None,
            "error": error,
            "timestamp": time.time()
        })
        self.touch()

    def log_error(self, error: str):
        """Append an error message to the error log"""
        self.error_log.append(error)
        self.touch()

    def increment_attempt(self):
        """Increment the attempt counter"""
        self.attempts += 1
        self.touch()

    def is_exhausted(self) -> bool:
        """Check if this cell has exceeded max retry attempts"""
        return self.attempts >= self.max_attempts

    def get_last_reflection(self) -> Optional[Dict[str, Any]]:
        """Return the most recent reflection or None"""
        return self.reflections[-1] if self.reflections else None

    def get_last_runtime_result(self) -> Optional[Dict[str, Any]]:
        """Return the most recent runtime result or None"""
        return self.runtime_results[-1] if self.runtime_results else None

    def get_failed_io_ops(self) -> List[Dict[str, Any]]:
        """Return all IO operations that failed"""
        return [r for r in self.io_history if not r["success"]]

    def get_failed_tools(self) -> List[Dict[str, Any]]:
        """Return all tool executions that failed"""
        return [r for r in self.tool_history if not r["success"]]

    def to_context_string(self) -> str:
        """Serialize the full cell state into a string for LLM context injection"""
        parts = []
        parts.append(f"Task ID: {self.task_id}")
        parts.append(f"Description: {self.task_description}")
        parts.append(f"Type: {self.task_type} | Priority: {self.task_priority} | Status: {self.task_status}")
        parts.append(f"Objective: {self.objective}")
        parts.append(f"Attempt: {self.attempts}/{self.max_attempts}")

        if self.problems:
            parts.append(f"\nProblems ({len(self.problems)}):")
            for i, p in enumerate(self.problems):
                parts.append(f"  [{i+1}] {p.get('description', str(p))}")

        if self.solutions:
            parts.append(f"\nSolutions ({len(self.solutions)}):")
            for i, s in enumerate(self.solutions):
                parts.append(f"  [{i+1}] {s.get('approach', str(s))}")

        if self.actions:
            parts.append(f"\nActions ({len(self.actions)}):")
            for i, a in enumerate(self.actions):
                parts.append(f"  [{i+1}] {a.get('action_plan', str(a))}")

        if self.reflections:
            last = self.reflections[-1]
            parts.append(f"\nLatest Reflection:")
            parts.append(f"  Status: {last.get('status', 'unknown')}")
            parts.append(f"  Feedback: {last.get('feedback', 'none')}")
            parts.append(f"  Rating: {last.get('rating', 'N/A')}/10")

        failed_io = self.get_failed_io_ops()
        if failed_io:
            parts.append(f"\nFailed IO Operations ({len(failed_io)}):")
            for f_io in failed_io:
                parts.append(f"  {f_io['operation']} on {f_io['file_path']}: {f_io.get('error', 'unknown')}")

        failed_tools = self.get_failed_tools()
        if failed_tools:
            parts.append(f"\nFailed Tools ({len(failed_tools)}):")
            for ft in failed_tools:
                parts.append(f"  {ft['tool_name']}: {ft.get('error', 'unknown')}")

        if self.error_log:
            parts.append(f"\nError Log:")
            for err in self.error_log[-5:]:
                parts.append(f"  {err}")

        return "\n".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the cell to a serializable dictionary"""
        return {
            "cell_id": self.cell_id,
            "task_id": self.task_id,
            "task_description": self.task_description,
            "task_type": self.task_type,
            "task_status": self.task_status,
            "task_priority": self.task_priority,
            "objective": self.objective,
            "dependencies": self.dependencies,
            "problems": self.problems,
            "solutions": self.solutions,
            "actions": self.actions,
            "reflections": self.reflections,
            "runtime_results": self.runtime_results,
            "io_history": self.io_history,
            "tool_history": self.tool_history,
            "error_log": self.error_log,
            "attempts": self.attempts,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }
