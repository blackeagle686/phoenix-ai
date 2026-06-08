from typing import Any, Dict, List, Optional
import time

from phoenix.framework.agent.memory.cell import MemoryCell
from phoenix.framework.agent.cognition.utils.id import generate_unique_id


class _SessionProxy:
    """Lightweight key value store for session variables"""

    def __init__(self):
        self._vars: Dict[str, Any] = {}

    def set(self, key: str, value: Any):
        self._vars[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self._vars.get(key, default)

    def get_all(self) -> Dict[str, Any]:
        return dict(self._vars)

    def clear(self):
        self._vars.clear()


class MemoryManager:
    """Central controller for the unified memory graph across the agent loop"""

    def __init__(self):
        self.cells: Dict[str, MemoryCell] = {}
        self.objective: str = ""
        self.session_id: str = ""
        self.task_order: List[str] = []
        self.global_context: Dict[str, Any] = {}
        self.completed_tasks: List[str] = []
        self.failed_tasks: List[str] = []
        self.session = _SessionProxy()
        self.interactions: List[Dict[str, Any]] = []

    def set_session(self, session_id: str, objective: str):
        """Bind a session and objective to this memory graph"""
        self.session_id = session_id
        self.objective = objective

    async def add_interaction(self, session_id: str, role: str, content: str, metadata: Optional[dict] = None):
        """Record an interaction in the conversation history"""
        self.interactions.append({
            "session_id": session_id,
            "role": role,
            "content": content,
            "metadata": metadata,
            "timestamp": time.time()
        })

    async def get_full_context(self, session_id: str = "", query: str = "") -> str:
        """Build context from interaction history for the planner initial call"""
        parts = []
        if self.objective:
            parts.append(f"OBJECTIVE: {self.objective}")

        recent = self.interactions[-20:]
        if recent:
            parts.append("\nCONVERSATION HISTORY:")
            for msg in recent:
                parts.append(f"  [{msg['role']}]: {msg['content'][:500]}")

        completed = self.get_completed_summary()
        if completed != "No tasks completed yet.":
            parts.append(f"\n{completed}")

        return "\n".join(parts) if parts else ""

    def create_cell(self, task_id: str, description: str, task_type: str = "other",
                    priority: str = "medium", dependencies: Optional[List[str]] = None) -> MemoryCell:
        """Create a new memory cell for a task and register it in the graph"""
        cell = MemoryCell(
            task_id=task_id,
            task_description=description,
            task_type=task_type,
            task_priority=priority,
            objective=self.objective,
            dependencies=dependencies or []
        )
        self.cells[task_id] = cell
        self.task_order.append(task_id)
        return cell

    def get_cell(self, task_id: str) -> MemoryCell:
        """Retrieve a cell by task id"""
        if task_id not in self.cells:
            raise ValueError(f"No memory cell for task {task_id}")
        return self.cells[task_id]

    def has_cell(self, task_id: str) -> bool:
        """Check if a cell exists for this task"""
        return task_id in self.cells

    def mark_complete(self, task_id: str):
        """Mark a task cell as done"""
        cell = self.get_cell(task_id)
        cell.task_status = "done"
        cell.touch()
        if task_id not in self.completed_tasks:
            self.completed_tasks.append(task_id)

    def mark_failed(self, task_id: str, error: str):
        """Mark a task cell as failed with an error"""
        cell = self.get_cell(task_id)
        cell.task_status = "failed"
        cell.log_error(error)
        if task_id not in self.failed_tasks:
            self.failed_tasks.append(task_id)

    def are_dependencies_met(self, task_id: str) -> bool:
        """Check if all dependency tasks are completed"""
        cell = self.get_cell(task_id)
        for dep_id in cell.dependencies:
            if dep_id not in self.completed_tasks:
                return False
        return True

    def get_pending_cells(self) -> List[MemoryCell]:
        """Return all cells still pending execution in order"""
        result = []
        for tid in self.task_order:
            cell = self.cells[tid]
            if cell.task_status in ("pending", "in_progress"):
                result.append(cell)
        return result

    def get_completed_summary(self) -> str:
        """Build a summary of all completed tasks for context"""
        parts = []
        for tid in self.completed_tasks:
            cell = self.cells[tid]
            parts.append(f"[DONE] {cell.task_id}: {cell.task_description}")
            if cell.reflections:
                last = cell.reflections[-1]
                parts.append(f"  Rating: {last.get('rating', 'N/A')}/10")
        return "\n".join(parts) if parts else "No tasks completed yet."

    def get_task_context(self, task_id: str) -> str:
        """Build full context string for the LLM including objective, completed work, and current cell"""
        parts = []
        parts.append(f"OBJECTIVE: {self.objective}")
        parts.append("")

        completed = self.get_completed_summary()
        if completed != "No tasks completed yet.":
            parts.append("COMPLETED TASKS:")
            parts.append(completed)
            parts.append("")

        cell = self.get_cell(task_id)
        parts.append("CURRENT TASK CONTEXT:")
        parts.append(cell.to_context_string())

        return "\n".join(parts)

    def get_io_context(self, task_id: str) -> str:
        """Build IO specific context showing file operations across the session"""
        parts = []
        parts.append("FILE IO HISTORY FOR THIS SESSION:")

        for tid in self.task_order:
            cell = self.cells[tid]
            if cell.io_history:
                parts.append(f"\n  Task {tid}:")
                for io_rec in cell.io_history:
                    status = "OK" if io_rec["success"] else "FAIL"
                    parts.append(f"    [{status}] {io_rec['operation']} {io_rec['file_path']}")

        current = self.get_cell(task_id)
        failed = current.get_failed_io_ops()
        if failed:
            parts.append(f"\nFAILED IO OPS IN CURRENT TASK:")
            for f_io in failed:
                parts.append(f"  {f_io['operation']} {f_io['file_path']}: {f_io.get('error')}")

        return "\n".join(parts)

    def get_tool_context(self, task_id: str) -> str:
        """Build tool specific context showing tool executions across the session"""
        parts = []
        parts.append("TOOL EXECUTION HISTORY:")

        current = self.get_cell(task_id)
        for t_rec in current.tool_history:
            status = "OK" if t_rec["success"] else "FAIL"
            parts.append(f"  [{status}] {t_rec['tool_name']}")
            if t_rec.get("error"):
                parts.append(f"    Error: {t_rec['error']}")

        return "\n".join(parts)

    def build_retry_context(self, task_id: str) -> str:
        """Build enriched context for retry attempts including all failure data"""
        cell = self.get_cell(task_id)
        parts = []
        parts.append(f"RETRY ATTEMPT {cell.attempts}/{cell.max_attempts}")
        parts.append("")

        parts.append(self.get_task_context(task_id))
        parts.append("")
        parts.append(self.get_io_context(task_id))
        parts.append("")
        parts.append(self.get_tool_context(task_id))

        return "\n".join(parts)

    def list_all_cells(self) -> List[MemoryCell]:
        """Return all cells in task order"""
        return [self.cells[tid] for tid in self.task_order if tid in self.cells]

    def clear(self):
        """Reset the entire memory graph"""
        self.cells.clear()
        self.task_order.clear()
        self.completed_tasks.clear()
        self.failed_tasks.clear()
        self.objective = ""
        self.global_context.clear()
        self.interactions.clear()
        self.session.clear()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the full memory graph"""
        return {
            "session_id": self.session_id,
            "objective": self.objective,
            "task_order": self.task_order,
            "completed_tasks": self.completed_tasks,
            "failed_tasks": self.failed_tasks,
            "cells": {tid: cell.to_dict() for tid, cell in self.cells.items()},
            "global_context": self.global_context
        }
