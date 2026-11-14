from typing import Any, Dict, List, Optional
import time

from phoenix.framework.agent.memory.cell import MemoryCell
from phoenix.framework.agent.memory.session import SessionMemory
from phoenix.framework.agent.memory.reflection import ReflectionMemory


class MemoryManager:
    """Central controller for the unified memory graph and conversation state across the agent loop."""

    def __init__(self, max_context_chars: int = 12000):
        self.cells: Dict[str, MemoryCell] = {}
        self.objective: str = ""
        self.session_id: str = ""
        self.task_order: List[str] = []
        self.global_context: Dict[str, Any] = {}
        self.completed_tasks: List[str] = []
        self.failed_tasks: List[str] = []
        self.session = SessionMemory()
        self.reflection = ReflectionMemory()
        self.interactions: List[Dict[str, Any]] = []
        self.max_context_chars = max_context_chars

    def set_session(self, session_id: str, objective: str):
        """Bind a session and objective to this memory graph. Resets task cells for a new run."""
        self.session_id = session_id
        self.objective = objective
        self.session.set("current_objective", objective)
        # Reset per-run execution graph while preserving conversation interactions and reflections
        self.cells.clear()
        self.task_order.clear()
        self.completed_tasks.clear()
        self.failed_tasks.clear()

    async def add_interaction(self, session_id: str, role: str, content: str, metadata: Optional[dict] = None):
        """Record an interaction in the conversation history."""
        if metadata and isinstance(metadata, dict):
            for k, v in metadata.items():
                self.session.set(k, v)
        self.interactions.append({
            "session_id": session_id,
            "role": role,
            "content": content,
            "metadata": metadata,
            "timestamp": time.time()
        })

    async def add_context_item(self, session_id: str, key: str, value: Any, source: str = "user"):
        """Store a structured context item in session variables and global context."""
        self.session.set(key, value)
        self.global_context[key] = {"value": value, "source": source, "session_id": session_id}

    async def get_context_bundle(self, session_id: str, query: str = "") -> Dict[str, Any]:
        """Return a structured context bundle with session variables and full text context."""
        full_text = await self.get_full_context(session_id=session_id, query=query)
        return {
            "session_id": session_id,
            "session_variables": self.session.get_all(),
            "global_context": dict(self.global_context),
            "full_text": full_text,
        }

    async def consolidate_reflections(self, llm: Any, session_id: Optional[str] = None):
        """Consolidate reflections using the provided LLM."""
        return await self.reflection.consolidate(llm)

    async def get_full_context(self, session_id: str = "", query: str = "") -> str:
        """Build context from interaction history, session variables, reflections, and completed tasks."""
        parts = []
        if self.objective:
            parts.append(f"OBJECTIVE: {self.objective}")

        session_vars = self.session.get_all()
        if session_vars:
            vars_str = ", ".join(f"{k}={v}" for k, v in session_vars.items())
            parts.append(f"SESSION VARIABLES: {vars_str}")

        filtered = (
            [m for m in self.interactions if not session_id or m.get("session_id") == session_id]
            if self.interactions
            else []
        )
        recent = filtered[-20:]
        if recent:
            parts.append("\nCONVERSATION HISTORY:")
            for msg in recent:
                role_label = str(msg.get("role", "user")).capitalize()
                content_str = str(msg.get("content", ""))[:800]
                parts.append(f"  {role_label}: {content_str}")

        reflections_str = self.reflection.get_reflections()
        if reflections_str:
            parts.append(f"\n{reflections_str}")

        completed = self.get_completed_summary()
        if completed != "No tasks completed yet.":
            parts.append(f"\n{completed}")

        full = "\n".join(parts) if parts else ""
        if len(full) > self.max_context_chars:
            return full[-self.max_context_chars :]
        return full

    def create_cell(
        self,
        task_id: str,
        description: str,
        task_type: str = "other",
        priority: str = "medium",
        dependencies: Optional[List[str]] = None,
    ) -> MemoryCell:
        """Create a new memory cell for a task and register it in the graph."""
        cell = MemoryCell(
            task_id=task_id,
            task_description=description,
            task_type=task_type,
            task_priority=priority,
            objective=self.objective,
            dependencies=dependencies or [],
        )
        self.cells[task_id] = cell
        if task_id not in self.task_order:
            self.task_order.append(task_id)
        return cell

    def get_cell(self, task_id: str) -> MemoryCell:
        """Retrieve a cell by task id."""
        if task_id not in self.cells:
            raise ValueError(f"No memory cell for task {task_id}")
        return self.cells[task_id]

    def has_cell(self, task_id: str) -> bool:
        """Check if a cell exists for this task."""
        return task_id in self.cells

    def mark_complete(self, task_id: str):
        """Mark a task cell as done."""
        cell = self.get_cell(task_id)
        cell.task_status = "done"
        cell.touch()
        if task_id not in self.completed_tasks:
            self.completed_tasks.append(task_id)

    def mark_failed(self, task_id: str, error: str):
        """Mark a task cell as failed with an error."""
        cell = self.get_cell(task_id)
        cell.task_status = "failed"
        cell.log_error(error)
        if task_id not in self.failed_tasks:
            self.failed_tasks.append(task_id)

    def are_dependencies_met(self, task_id: str) -> bool:
        """Check if all dependency tasks that exist in the plan are completed."""
        cell = self.get_cell(task_id)
        for dep_id in cell.dependencies:
            if dep_id == task_id:
                continue
            # Only block on dependencies that actually exist in the current plan
            if dep_id in self.cells and dep_id not in self.completed_tasks:
                return False
        return True

    def get_pending_cells(self) -> List[MemoryCell]:
        """Return all cells still pending execution in order."""
        result = []
        for tid in self.task_order:
            cell = self.cells[tid]
            if cell.task_status in ("pending", "in_progress"):
                result.append(cell)
        return result

    def get_completed_summary(self) -> str:
        """Build a summary of all completed tasks for context, including key tool outputs."""
        parts = []
        for tid in self.completed_tasks:
            cell = self.cells[tid]
            parts.append(f"[DONE] {cell.task_id}: {cell.task_description}")
            for t_rec in cell.get_successful_tools()[-3:]:
                if t_rec.get("output"):
                    parts.append(f"  Tool ({t_rec['tool_name']}) Output: {str(t_rec['output'])[:600]}")
            if cell.reflections:
                last = cell.reflections[-1]
                parts.append(f"  Rating: {last.get('rating', 'N/A')}/10")
                if last.get("summary"):
                    parts.append(f"  Summary: {last['summary']}")
        return "\n".join(parts) if parts else "No tasks completed yet."

    def get_execution_digest(self) -> str:
        """Build a complete digest of all task results and outputs for final answer synthesis."""
        parts = []
        for tid in self.task_order:
            cell = self.cells.get(tid)
            if not cell:
                continue
            parts.append(f"Task [{cell.task_status.upper()}] {cell.task_id}: {cell.task_description}")
            for t_rec in cell.tool_history:
                status = "OK" if t_rec["success"] else "FAIL"
                parts.append(f"  - Tool {t_rec['tool_name']} [{status}]: {str(t_rec.get('output') or t_rec.get('error') or '')[:1500]}")
            for r_res in cell.runtime_results[-2:]:
                res_payload = r_res.get("result") if isinstance(r_res, dict) else None
                if isinstance(res_payload, dict) and res_payload.get("direct_response"):
                    parts.append(f"  - Direct Finding: {res_payload['direct_response']}")
            if cell.reflections:
                last = cell.reflections[-1]
                if last.get("feedback"):
                    parts.append(f"  - Reflection: {last['feedback']} (Rating: {last.get('rating', 'N/A')}/10)")
        return "\n".join(parts)

    def get_task_context(self, task_id: str) -> str:
        """Build full context string for the LLM including objective, completed work, and current cell."""
        parts = [f"OBJECTIVE: {self.objective}", ""]

        completed = self.get_completed_summary()
        if completed != "No tasks completed yet.":
            parts.append("COMPLETED TASKS:")
            parts.append(completed)
            parts.append("")

        cell = self.get_cell(task_id)
        parts.append("CURRENT TASK CONTEXT:")
        parts.append(cell.to_context_string())

        full = "\n".join(parts)
        if len(full) > self.max_context_chars:
            return full[-self.max_context_chars :]
        return full

    def get_io_context(self, task_id: str) -> str:
        """Build IO specific context showing file operations across the session."""
        parts = ["FILE IO HISTORY FOR THIS SESSION:"]

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
            parts.append("\nFAILED IO OPS IN CURRENT TASK:")
            for f_io in failed:
                parts.append(f"  {f_io['operation']} {f_io['file_path']}: {f_io.get('error')}")

        return "\n".join(parts)

    def get_tool_context(self, task_id: str) -> str:
        """Build tool specific context showing tool executions across the session."""
        parts = ["TOOL EXECUTION HISTORY:"]

        current = self.get_cell(task_id)
        for t_rec in current.tool_history:
            status = "OK" if t_rec["success"] else "FAIL"
            parts.append(f"  [{status}] {t_rec['tool_name']}")
            if t_rec.get("output"):
                parts.append(f"    Output: {str(t_rec['output'])[:800]}")
            if t_rec.get("error"):
                parts.append(f"    Error: {t_rec['error']}")

        return "\n".join(parts)

    def build_retry_context(self, task_id: str) -> str:
        """Build enriched context for retry attempts including all execution and failure data."""
        cell = self.get_cell(task_id)
        parts = [
            f"ATTEMPT {cell.attempts}/{cell.max_attempts}",
            "",
            self.get_task_context(task_id),
            "",
            self.get_io_context(task_id),
            "",
            self.get_tool_context(task_id),
        ]
        full = "\n".join(parts)
        if len(full) > self.max_context_chars:
            return full[-self.max_context_chars :]
        return full

    def list_all_cells(self) -> List[MemoryCell]:
        """Return all cells in task order."""
        return [self.cells[tid] for tid in self.task_order if tid in self.cells]

    def clear(self):
        """Reset the entire memory graph."""
        self.cells.clear()
        self.task_order.clear()
        self.completed_tasks.clear()
        self.failed_tasks.clear()
        self.objective = ""
        self.global_context.clear()
        self.interactions.clear()
        self.session.clear()
        self.reflection.clear()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the full memory graph."""
        return {
            "session_id": self.session_id,
            "objective": self.objective,
            "task_order": self.task_order,
            "completed_tasks": self.completed_tasks,
            "failed_tasks": self.failed_tasks,
            "cells": {tid: cell.to_dict() for tid, cell in self.cells.items()},
            "global_context": self.global_context,
        }
