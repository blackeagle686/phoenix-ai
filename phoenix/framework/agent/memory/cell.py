from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import time

from phoenix.framework.agent.cognition.utils.id import generate_unique_id
from phoenix.framework.agent.memory.budget import compact_value, truncate_text


def _append_bounded(items: List[Any], entry: Any, max_len: int) -> None:
    items.append(entry)
    if len(items) > max_len:
        del items[: len(items) - max_len]


@dataclass
class MemoryCell:
    """Bounded Working Memory node holding execution context for one task lifecycle."""

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
    max_history: int = 8
    max_tool_output_chars: int = 2400
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    _compacted: bool = False

    def touch(self):
        """Update the timestamp."""
        self.updated_at = time.time()

    def add_problem(self, problem: Dict[str, Any]):
        """Push a bounded problem definition into this cell."""
        _append_bounded(self.problems, problem, self.max_history)
        self.touch()

    def add_solution(self, solution: Dict[str, Any]):
        """Push a bounded solution definition into this cell."""
        _append_bounded(self.solutions, solution, self.max_history)
        self.touch()

    def add_action(self, action: Dict[str, Any]):
        """Push a bounded action record into this cell."""
        _append_bounded(self.actions, action, self.max_history)
        self.touch()

    def add_reflection(self, reflection: Dict[str, Any]):
        """Push a bounded reflection record into this cell."""
        _append_bounded(self.reflections, reflection, 4)
        self.touch()

    def add_runtime_result(self, result: Dict[str, Any]):
        """
        Store a deduplicated runtime summary into this cell.
        Detailed tool outputs belong in `tool_history`, avoiding double memory bloat.
        """
        if isinstance(result, dict):
            compact_rec: Dict[str, Any] = {
                "task_id": result.get("task_id", self.task_id),
                "tool_name": result.get("tool_name"),
                "success": result.get("success", True),
            }
            if result.get("error_context"):
                compact_rec["error_context"] = truncate_text(str(result["error_context"]), 300)
            payload = result.get("result")
            if isinstance(payload, dict):
                compact_payload: Dict[str, Any] = {}
                if payload.get("direct_response"):
                    compact_payload["direct_response"] = truncate_text(str(payload["direct_response"]), 1200)
                if payload.get("thought_process"):
                    compact_payload["thought_process"] = truncate_text(str(payload["thought_process"]), 400)
                if "execution_results" in payload:
                    compact_payload["execution_count"] = len(payload.get("execution_results") or [])
                compact_rec["result"] = compact_payload
            else:
                compact_rec["result"] = compact_value(payload, max_chars=800) if payload is not None else None
        else:
            compact_rec = {"output": compact_value(result, max_chars=800)}

        _append_bounded(self.runtime_results, compact_rec, 4)
        self.touch()

    def add_io_record(self, operation: str, file_path: str, success: bool, error: Optional[str] = None):
        """Record a bounded file IO operation that was executed."""
        _append_bounded(
            self.io_history,
            {
                "operation": operation,
                "file_path": file_path,
                "success": success,
                "error": truncate_text(str(error), 250) if error else None,
                "timestamp": time.time(),
            },
            self.max_history,
        )
        self.touch()

    def add_tool_record(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        success: bool,
        output: Any = None,
        error: Optional[str] = None,
    ):
        """Record a bounded, compact tool execution."""
        compact_args = {
            k: truncate_text(str(v), 200) if isinstance(v, str) and len(v) > 200 else v
            for k, v in (arguments or {}).items()
        }
        _append_bounded(
            self.tool_history,
            {
                "tool_name": tool_name,
                "arguments": compact_args,
                "success": success,
                "output": compact_value(output, max_chars=self.max_tool_output_chars) if output is not None else None,
                "error": truncate_text(str(error), 350) if error else None,
                "timestamp": time.time(),
            },
            self.max_history,
        )
        self.touch()

    def log_error(self, error: str):
        """Append a bounded error message to the error log."""
        msg = truncate_text(str(error), 400)
        if not self.error_log or self.error_log[-1] != msg:
            _append_bounded(self.error_log, msg, 5)
        self.touch()

    def increment_attempt(self):
        """Increment the attempt counter."""
        self.attempts += 1
        self.touch()

    def is_exhausted(self) -> bool:
        """Check if this cell has exceeded max retry attempts."""
        return self.attempts >= self.max_attempts

    def get_last_reflection(self) -> Optional[Dict[str, Any]]:
        return self.reflections[-1] if self.reflections else None

    def get_last_runtime_result(self) -> Optional[Dict[str, Any]]:
        return self.runtime_results[-1] if self.runtime_results else None

    def get_failed_io_ops(self) -> List[Dict[str, Any]]:
        return [r for r in self.io_history if not r["success"]]

    def get_failed_tools(self) -> List[Dict[str, Any]]:
        return [r for r in self.tool_history if not r["success"]]

    def get_successful_tools(self) -> List[Dict[str, Any]]:
        return [r for r in self.tool_history if r["success"]]

    def get_direct_findings(self) -> List[str]:
        findings: List[str] = []
        for rr in self.runtime_results:
            if isinstance(rr, dict):
                res_payload = rr.get("result")
                if isinstance(res_payload, dict) and res_payload.get("direct_response"):
                    findings.append(str(res_payload["direct_response"]))
        return findings

    def get_outcome_summary(self, max_chars: int = 500) -> str:
        """Return a dense summary of what this task accomplished (findings, tools, or reflection)."""
        parts: List[str] = []
        findings = self.get_direct_findings()
        if findings:
            parts.append(f"Finding: {truncate_text(findings[-1], 320)}")

        ok_tools = self.get_successful_tools()
        for t_rec in ok_tools[-2:]:
            out = t_rec.get("output") or ""
            if out:
                parts.append(f"{t_rec['tool_name']} -> {truncate_text(str(out), 260)}")

        if self.reflections:
            last = self.reflections[-1]
            summary = last.get("summary") or last.get("feedback")
            if summary:
                parts.append(f"Summary: {truncate_text(str(summary), 200)}")

        if not parts and self.error_log:
            parts.append(f"Errors: {truncate_text(self.error_log[-1], 220)}")

        return truncate_text(" | ".join(parts), max_chars=max_chars)

    def compact(self) -> None:
        """
        Reclaim working RAM once the task finishes: shrink large tool outputs and
        drop transient problem/solution/runtime scaffolding while preserving outcomes.
        """
        if self._compacted:
            return
        self.problems = self.problems[-1:]
        self.solutions = self.solutions[-1:]
        self.actions = self.actions[-2:]
        for rec in self.tool_history:
            if rec.get("output") and len(str(rec["output"])) > 400:
                rec["output"] = truncate_text(str(rec["output"]), 400, mode="middle")
        self.runtime_results = self.runtime_results[-1:]
        self._compacted = True

    def to_context_string(self, max_chars: int = 2400) -> str:
        """Serialize the current working cell into a concise, non-redundant prompt slice."""
        parts = [
            f"Task [{self.task_id}] ({self.task_status}, attempt {self.attempts}/{self.max_attempts}): {self.task_description}"
        ]

        if self.tool_history:
            parts.append("Recent Tools:")
            for t_rec in self.tool_history[-3:]:
                status = "OK" if t_rec["success"] else "FAIL"
                args_str = compact_value(t_rec.get("arguments", {}), max_chars=140)
                parts.append(f"  [{status}] {t_rec['tool_name']}({args_str})")
                if t_rec.get("output"):
                    parts.append(f"    -> {truncate_text(str(t_rec['output']), 700, mode='middle')}")
                if t_rec.get("error"):
                    parts.append(f"    err: {truncate_text(str(t_rec['error']), 220)}")

        findings = self.get_direct_findings()
        if findings:
            parts.append(f"Direct Finding: {truncate_text(findings[-1], 360)}")

        if self.reflections:
            last = self.reflections[-1]
            parts.append(
                f"Last Feedback ({last.get('rating', 'N/A')}/10): {truncate_text(str(last.get('feedback', '')), 280)}"
            )

        failed_io = self.get_failed_io_ops()
        if failed_io:
            parts.append("Failed IO: " + "; ".join(f"{f['operation']} {f['file_path']} ({f.get('error', '')})" for f in failed_io[-2:]))

        if self.error_log and not (self.tool_history and self.get_failed_tools()):
            parts.append(f"Latest Error: {self.error_log[-1]}")

        return truncate_text("\n".join(parts), max_chars=max_chars, mode="tail_ellipsis")

    def to_dict(self) -> Dict[str, Any]:
        """Convert the cell to a serializable dictionary."""
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
            "updated_at": self.updated_at,
        }
