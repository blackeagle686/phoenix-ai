from typing import Any, Dict, List, Optional

from phoenix.framework.agent.memory.budget import (
    compact_value,
    estimate_tokens,
    fit_sections,
    truncate_text,
)
from phoenix.framework.agent.memory.cell import MemoryCell
from phoenix.framework.agent.memory.session import SessionMemory
from phoenix.framework.agent.memory.reflection import ReflectionMemory
from phoenix.framework.agent.memory.short_term.stm_manager import ShortTermMemoryManager
from phoenix.framework.agent.memory.long_term.ltm_manager import LongTermMemoryManager


class MemoryManager:
    """
    Optimized 3-Tier Context Memory Orchestrator:
      1. Working Memory (`MemoryCell` DAG graph): Bounded per-run task state, compacted after completion.
      2. Short-Term Memory (`ShortTermMemoryManager`): Bounded sliding-window session chat + rolling summary.
      3. Long-Term Memory (`LongTermMemoryManager`): Selective, deduplicated, relevance-scored facts/lessons.
    """

    def __init__(
        self,
        max_context_chars: int = 8000,
        max_context_tokens: int = 1800,
        stm_window: int = 12,
        ltm_max_items: int = 200,
        ltm_storage_path: Optional[str] = None,
    ):
        self.max_context_chars = max(400, max_context_chars)
        self.max_context_tokens = min(max(100, max_context_tokens), max(100, self.max_context_chars // 4))

        self.cells: Dict[str, MemoryCell] = {}
        self.objective: str = ""
        self.session_id: str = ""
        self.task_order: List[str] = []
        self.global_context: Dict[str, Any] = {}
        self.completed_tasks: List[str] = []
        self.failed_tasks: List[str] = []

        self.short_term = ShortTermMemoryManager(max_cells=stm_window)
        self.long_term = LongTermMemoryManager(max_items=ltm_max_items, storage_path=ltm_storage_path)
        self.session = SessionMemory()
        self.reflection = ReflectionMemory()

    @property
    def interactions(self) -> List[Dict[str, Any]]:
        """Derived bounded view of recent short-term interactions (zero duplicate storage)."""
        items: List[Dict[str, Any]] = []
        for sid, bucket in self.short_term._sessions.items():
            for cell in bucket:
                items.append({
                    "session_id": sid,
                    "role": cell.role,
                    "content": cell.content,
                    "metadata": cell.metadata,
                    "timestamp": cell.timestamp,
                })
        return items

    def set_session(self, session_id: str, objective: str):
        """Bind a session and objective for a new execution run and reset transient working cells."""
        self.session_id = session_id
        self.objective = objective
        self.session.set("current_objective", objective)
        self.cells.clear()
        self.task_order.clear()
        self.completed_tasks.clear()
        self.failed_tasks.clear()

    async def add_interaction(self, session_id: str, role: str, content: str, metadata: Optional[dict] = None):
        """Record a user/assistant interaction into bounded Short-Term Memory (filtering noisy system logs)."""
        if metadata and isinstance(metadata, dict):
            for k, v in metadata.items():
                if k != "persist":
                    self.session.set(k, v)

        norm_role = str(role or "user").lower()
        # Filter internal loop progress system messages from conversational STM unless explicitly persisted
        if norm_role == "system" and not (isinstance(metadata, dict) and metadata.get("persist")):
            return

        clean_content = str(content or "")
        # Strip verbose execution summary footers before saving assistant turns to STM
        marker = "\n\n---\nExecution Summary:\n"
        if norm_role == "assistant" and marker in clean_content:
            clean_content = clean_content.split(marker, 1)[0].strip()

        await self.short_term.add(
            session_id or self.session_id or "default",
            {
                "role": norm_role,
                "content": clean_content,
                "metadata": metadata or {},
            },
        )

    async def add_context_item(
        self,
        session_id: str,
        key: str,
        value: Any,
        source: str = "user",
        long_term: bool = True,
        pinned: bool = False,
    ):
        """Store a structured context item into session state and optionally Long-Term Memory."""
        self.session.set(key, value)
        self.global_context[key] = {"value": value, "source": source, "session_id": session_id}
        if long_term:
            self.long_term.add(
                content=f"{key}: {compact_value(value, max_chars=240)}",
                memory_type="fact",
                session_id=session_id,
                importance=0.85 if pinned else 0.75,
                metadata={"pinned": pinned, "key": key, "source": source},
            )

    async def get_context_bundle(self, session_id: str, query: str = "") -> Dict[str, Any]:
        """Return a structured context bundle with session variables and token-budgeted full text."""
        full_text = await self.get_full_context(session_id=session_id, query=query)
        return {
            "session_id": session_id,
            "session_variables": self.session.get_all(),
            "global_context": dict(self.global_context),
            "full_text": full_text,
            "stats": self.get_stats(),
        }

    async def consolidate_reflections(self, llm: Any, session_id: Optional[str] = None):
        """Consolidate reflections via LLM and promote the distilled insight to Long-Term Memory."""
        insight = await self.reflection.consolidate(llm)
        if insight:
            self.long_term.add(
                content=insight,
                memory_type="lesson",
                session_id=session_id or self.session_id,
                importance=0.8,
            )
        return insight

    async def _get_stm_context(self, session_id: str, max_chars: int = 2000) -> str:
        if session_id:
            return await self.short_term.get_context_string(session_id, max_chars=max_chars)
        # Fallback if session_id was omitted: inspect most recently active session
        if self.short_term._sessions:
            last_sid = next(reversed(self.short_term._sessions))
            return await self.short_term.get_context_string(last_sid, max_chars=max_chars)
        return ""

    async def get_full_context(
        self,
        session_id: str = "",
        query: str = "",
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Assemble an accurate, strictly token-budgeted context across all active memory tiers.
        """
        budget_tokens = max_tokens or self.max_context_tokens
        sections = []

        if self.objective:
            sections.append((0, "OBJECTIVE:", self.objective))

        session_vars_str = self.session.to_context_string(
            exclude_keys=("current_objective",) if self.objective else (),
            max_chars=450,
        )
        if session_vars_str:
            sections.append((1, "SESSION VARIABLES:", session_vars_str))

        stm_str = await self._get_stm_context(session_id, max_chars=min(2400, budget_tokens * 2))
        if stm_str:
            sections.append((1, "CONVERSATION HISTORY:", stm_str))

        if self.long_term.count() > 0:
            # Exclude session var duplicates by skipping exact matching keys when session vars exist
            ltm_items = self.long_term.retrieve(query=query, limit=4, session_id=session_id or None)
            session_keys = set(self.session.get_all().keys())
            filtered_ltm = [
                c for c in ltm_items
                if c.metadata.get("key") not in session_keys
            ]
            if filtered_ltm:
                ltm_str = "\n".join(f"- [{c.memory_type}] {c.content}" for c in filtered_ltm[:3])
                sections.append((2, "LONG-TERM MEMORY:", truncate_text(ltm_str, 650)))

        reflections_str = self.reflection.get_reflections(limit=4, max_chars=550)
        if reflections_str:
            sections.append((2, "", reflections_str))

        completed = self.get_completed_summary(brief=True)
        if completed != "No tasks completed yet.":
            sections.append((3, "COMPLETED TASKS:", completed))

        return fit_sections(sections, max_tokens=budget_tokens)

    def create_cell(
        self,
        task_id: str,
        description: str,
        task_type: str = "other",
        priority: str = "medium",
        dependencies: Optional[List[str]] = None,
    ) -> MemoryCell:
        """Create a new working memory cell for a task and register it in the graph."""
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
        if task_id not in self.cells:
            raise ValueError(f"No memory cell for task {task_id}")
        return self.cells[task_id]

    def has_cell(self, task_id: str) -> bool:
        return task_id in self.cells

    def mark_complete(self, task_id: str):
        cell = self.get_cell(task_id)
        cell.task_status = "done"
        cell.touch()
        if task_id not in self.completed_tasks:
            self.completed_tasks.append(task_id)

    def mark_failed(self, task_id: str, error: str):
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
            if dep_id in self.cells and dep_id not in self.completed_tasks:
                return False
        return True

    def get_pending_cells(self) -> List[MemoryCell]:
        return [
            self.cells[tid]
            for tid in self.task_order
            if self.cells[tid].task_status in ("pending", "in_progress")
        ]

    def get_completed_summary(self, brief: bool = False) -> str:
        """Build a token-efficient summary of completed tasks."""
        if not self.completed_tasks:
            return "No tasks completed yet."
        parts = []
        for tid in self.completed_tasks:
            cell = self.cells.get(tid)
            if not cell:
                continue
            if brief:
                outcome = cell.get_outcome_summary(max_chars=160)
                line = f"[DONE] {cell.task_id}: {truncate_text(cell.task_description, 90)}"
                if outcome:
                    line += f" => {outcome}"
                parts.append(line)
            else:
                parts.append(f"[DONE] {cell.task_id}: {cell.task_description}")
                outcome = cell.get_outcome_summary(max_chars=380)
                if outcome:
                    parts.append(f"  Result: {outcome}")
        return "\n".join(parts) if parts else "No tasks completed yet."

    def get_execution_digest(self, max_tokens: Optional[int] = None) -> str:
        """Build a token-budgeted digest of all executed tasks and outcomes for final answer synthesis."""
        budget_tokens = max_tokens or self.max_context_tokens
        sections = []
        for tid in self.task_order:
            cell = self.cells.get(tid)
            if not cell:
                continue
            lines = []
            for t_rec in cell.tool_history[-3:]:
                status = "OK" if t_rec["success"] else "FAIL"
                detail = t_rec.get("output") or t_rec.get("error") or ""
                lines.append(f"  - Tool {t_rec['tool_name']} [{status}]: {truncate_text(str(detail), 600, mode='middle')}")
            for finding in cell.get_direct_findings()[-1:]:
                lines.append(f"  - Finding: {truncate_text(finding, 500)}")
            if cell.reflections:
                last = cell.reflections[-1]
                if last.get("summary") or last.get("feedback"):
                    msg = last.get("summary") or last.get("feedback")
                    lines.append(f"  - Reflection ({last.get('rating', 'N/A')}/10): {truncate_text(str(msg), 220)}")

            prio = 1 if cell.task_status in ("done", "failed") else 2
            header = f"Task [{cell.task_status.upper()}] {cell.task_id}: {cell.task_description}"
            sections.append((prio, header, "\n".join(lines) or "  - No tool outputs recorded."))

        return fit_sections(sections, max_tokens=budget_tokens)

    def get_task_context(self, task_id: str, max_tokens: Optional[int] = None) -> str:
        """
        Build a focused, dependency-aware context for solving `task_id`.
        Direct dependencies receive full output details; unrelated tasks get 1-line summaries.
        """
        budget_tokens = max_tokens or self.max_context_tokens
        cell = self.get_cell(task_id)
        dep_ids = set(cell.dependencies or ())

        sections = [
            (0, "OBJECTIVE:", self.objective or cell.objective),
            (0, "CURRENT TASK CONTEXT:", cell.to_context_string(max_chars=2200)),
        ]

        # Priority 1: Exact outputs from direct upstream dependencies
        if dep_ids:
            dep_lines = []
            for d_id in cell.dependencies:
                dep_cell = self.cells.get(d_id)
                if not dep_cell:
                    continue
                dep_lines.append(f"[{dep_cell.task_status.upper()}] {d_id}: {dep_cell.task_description}")
                outcome = dep_cell.get_outcome_summary(max_chars=800)
                if outcome:
                    dep_lines.append(f"  Output: {outcome}")
            if dep_lines:
                sections.append((1, "DEPENDENCY OUTPUTS:", "\n".join(dep_lines)))

        # Priority 2: Compact session vars (if non-empty)
        sess_str = self.session.to_context_string(exclude_keys=("current_objective",), max_chars=350)
        if sess_str:
            sections.append((2, "SESSION STATE:", sess_str))

        # Priority 3: 1-line awareness of other completed tasks
        other_done = [
            tid for tid in self.completed_tasks
            if tid not in dep_ids and tid != task_id and tid in self.cells
        ]
        if other_done:
            brief_lines = [
                f"- [DONE] {tid}: {truncate_text(self.cells[tid].task_description, 80)}"
                for tid in other_done
            ]
            sections.append((3, "OTHER COMPLETED TASKS:", "\n".join(brief_lines)))

        return fit_sections(sections, max_tokens=budget_tokens)

    def get_io_context(self, task_id: str) -> str:
        """Build compact IO history across current run."""
        parts = ["FILE IO HISTORY:"]
        for tid in self.task_order:
            cell = self.cells[tid]
            if cell.io_history:
                ops = ", ".join(
                    f"[{'OK' if r['success'] else 'FAIL'}] {r['operation']} {r['file_path']}"
                    for r in cell.io_history[-4:]
                )
                parts.append(f"  {tid}: {ops}")

        current = self.get_cell(task_id)
        failed = current.get_failed_io_ops()
        if failed:
            parts.append("FAILED IO OPS:")
            for f_io in failed[-3:]:
                parts.append(f"  {f_io['operation']} {f_io['file_path']}: {f_io.get('error')}")
        return "\n".join(parts)

    def get_tool_context(self, task_id: str) -> str:
        """Build compact tool execution history for `task_id`."""
        parts = ["TOOL EXECUTION HISTORY:"]
        current = self.get_cell(task_id)
        for t_rec in current.tool_history[-5:]:
            status = "OK" if t_rec["success"] else "FAIL"
            parts.append(f"  [{status}] {t_rec['tool_name']}")
            if t_rec.get("output"):
                parts.append(f"    Output: {truncate_text(str(t_rec['output']), 700, mode='middle')}")
            if t_rec.get("error"):
                parts.append(f"    Error: {t_rec['error']}")
        return "\n".join(parts)

    def build_retry_context(self, task_id: str, max_tokens: Optional[int] = None) -> str:
        """
        Build a non-redundant reflection & retry slice for `task_id`.
        Eliminates duplicate concatenation of tool/IO histories for significant token savings.
        """
        budget_tokens = max_tokens or min(1200, self.max_context_tokens)
        cell = self.get_cell(task_id)

        sections = [
            (0, f"ATTEMPT {cell.attempts}/{cell.max_attempts} | OBJECTIVE:", self.objective or cell.objective),
            (0, "TASK EXECUTION SLICE:", cell.to_context_string(max_chars=2400)),
        ]
        failed_io = cell.get_failed_io_ops()
        if failed_io:
            io_err = "\n".join(
                f"  - {r['operation']} {r['file_path']}: {r.get('error')}"
                for r in failed_io[-3:]
            )
            sections.append((1, "FAILED IO DETAILS:", io_err))

        return fit_sections(sections, max_tokens=budget_tokens)

    async def finalize_run(
        self,
        session_id: str,
        objective: str = "",
        final_answer: str = "",
        llm: Any = None,
    ) -> None:
        """
        Post-run maintenance:
          1. Selectively promote high-signal run lessons/outcomes into Long-Term Memory.
          2. Auto-consolidate reflections when threshold is reached.
          3. Compact finished working memory cells to keep RAM minimal.
        """
        obj = objective or self.objective
        if obj:
            if self.failed_tasks:
                failed_notes = "; ".join(
                    f"{tid}: {(self.cells[tid].error_log[-1] if tid in self.cells and self.cells[tid].error_log else 'failed')}"
                    for tid in self.failed_tasks[:2]
                )
                self.long_term.add(
                    content=f"Goal '{truncate_text(obj, 90)}' hit errors ({truncate_text(failed_notes, 180)}).",
                    memory_type="lesson",
                    session_id=session_id,
                    importance=0.75,
                )
            elif self.completed_tasks:
                done_list = ", ".join(self.completed_tasks[:4])
                takeaway = truncate_text(final_answer.replace("\n", " "), 180) if final_answer else ""
                note = f"Completed '{truncate_text(obj, 90)}' via [{done_list}]"
                if takeaway:
                    note += f": {takeaway}"
                self.long_term.add(
                    content=note,
                    memory_type="outcome",
                    session_id=session_id,
                    importance=0.6,
                )

        if len(self.reflection.reflections) >= self.reflection.consolidation_threshold:
            if llm is not None and hasattr(llm, "generate"):
                await self.consolidate_reflections(llm, session_id=session_id)
            else:
                self.reflection.reflections = self.reflection.reflections[-3:]

        for cell in self.cells.values():
            if cell.task_status in ("done", "failed", "skipped"):
                cell.compact()

    def list_all_cells(self) -> List[MemoryCell]:
        return [self.cells[tid] for tid in self.task_order if tid in self.cells]

    def get_stats(self) -> Dict[str, Any]:
        """Return memory footprint & token efficiency metrics."""
        return {
            "working_cells": len(self.cells),
            "completed_tasks": len(self.completed_tasks),
            "failed_tasks": len(self.failed_tasks),
            "stm_active_turns": self.short_term.count(),
            "stm_evicted_summarized": self.short_term.total_evicted(),
            "ltm_items": self.long_term.count(),
            "reflections": len(self.reflection.reflections),
            "session_vars": len(self.session.get_all()),
            "max_context_tokens": self.max_context_tokens,
        }

    def clear(self):
        """Reset all memory tiers."""
        self.cells.clear()
        self.task_order.clear()
        self.completed_tasks.clear()
        self.failed_tasks.clear()
        self.objective = ""
        self.global_context.clear()
        self.short_term.clear()
        self.long_term.clear()
        self.session.clear()
        self.reflection.clear()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the current memory graph and tier stats."""
        return {
            "session_id": self.session_id,
            "objective": self.objective,
            "task_order": self.task_order,
            "completed_tasks": self.completed_tasks,
            "failed_tasks": self.failed_tasks,
            "cells": {tid: cell.to_dict() for tid, cell in self.cells.items()},
            "global_context": self.global_context,
            "stats": self.get_stats(),
        }


ContextMemory = MemoryManager
