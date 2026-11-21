from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

from .base import BaseActor
from phoenix.framework.agent.cognition.schema import ActorInputSchema, ActorOutputSchema
from phoenix.framework.agent.memory.budget import compact_value
from phoenix.framework.agent.tools._fs_utils import check_runtime_path

if TYPE_CHECKING:
    from phoenix.framework.agent.cognition.schema import ActionSchema, ProblemSchema, SolutionSchema


READ_ONLY_TOOL_NAMES = frozenset({
    "file_read",
    "file_search",
    "python_analyzer",
    "compile_code",
    "web_search",
    "wikipedia_search",
    "arxiv_search",
    "web_scraper",
})

MUTATING_FILE_TOOLS = frozenset({
    "file_write",
    "file_edit",
    "file_update_multi",
    "file_append",
})


class Actor(BaseActor):
    """Streamlined, deterministic tool execution engine for the Thinker and AgentLoop."""

    def __init__(
        self,
        tool_manager: Any,
        thinker: Optional[Any] = None,
        runtime: Optional[Any] = None,
        reflector: Optional[Any] = None,
    ):
        super().__init__(tool_manager=tool_manager, thinker=thinker, reflector=reflector, runtime=runtime)
        self._read_files: Set[str] = set()

    def reset_session_tracking(self) -> None:
        """Reset tracked read files at the start of a new top-level run when desired."""
        self._read_files.clear()

    def _active_runtime(self) -> Any:
        return getattr(self.tool_manager, "runtime", None) or self.runtime

    def _get_registered_tool(self, tool_name: str) -> Optional[Any]:
        if not self.tool_manager:
            return None
        if hasattr(self.tool_manager, "get_tool") and callable(self.tool_manager.get_tool):
            return self.tool_manager.get_tool(tool_name)
        registry = getattr(self.tool_manager, "registry", None)
        tools_map = getattr(registry, "tools", None)
        return tools_map.get(tool_name) if isinstance(tools_map, dict) else None

    def _is_read_only_tool(self, tool_name: str) -> bool:
        if tool_name in READ_ONLY_TOOL_NAMES:
            return True
        tool_obj = self._get_registered_tool(tool_name)
        return bool(getattr(tool_obj, "read_only", False)) if tool_obj is not None else False

    def _is_return_direct_tool(self, tool_name: str) -> bool:
        tool_obj = self._get_registered_tool(tool_name)
        return bool(getattr(tool_obj, "return_direct", False)) if tool_obj is not None else False

    def _preflight_guard(self, tool_name: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Prevent blind full-file overwrites of existing non-trivial files before reading them."""
        if tool_name != "file_write":
            return None
        from_line = payload.get("from_line", 1) or 1
        to_line = payload.get("to_line")
        if int(from_line) > 1 or to_line is not None:
            return None

        raw_path = payload.get("file_path") or payload.get("path")
        if not raw_path:
            return None

        rt = self._active_runtime()
        if not rt or not getattr(rt, "workspace_root", None):
            return None

        ok, abs_path, _ = check_runtime_path(rt, str(raw_path))
        if not ok or not os.path.isfile(abs_path):
            return None

        real_path = os.path.realpath(abs_path)
        if real_path in self._read_files:
            return None

        try:
            with open(real_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.read().splitlines()
        except OSError:
            return None

        if len(lines) >= 4:
            return {
                "tool": tool_name,
                "arguments": payload,
                "success": False,
                "output": {"file_path": abs_path, "existing_lines": len(lines)},
                "error": (
                    f"Safety Guard: '{os.path.basename(abs_path)}' already exists ({len(lines)} lines) "
                    "and has not been inspected yet. Call `file_read` first to read its content, "
                    "then update it safely with `file_edit` (or call `file_write` again if a full rewrite is intended)."
                ),
            }
        return None

    def _track_path_access(self, tool_name: str, res_dict: Dict[str, Any]) -> None:
        if not res_dict.get("success"):
            return
        out = res_dict.get("output")
        if not isinstance(out, dict):
            return
        fpath = out.get("file_path")
        if not fpath or out.get("is_directory"):
            return
        if tool_name in ("file_read", "python_analyzer") or tool_name in MUTATING_FILE_TOOLS:
            self._read_files.add(os.path.realpath(str(fpath)))

    def _record_side_effects(self, tool_name: str, rec: Dict[str, Any], artifacts: Dict[str, Any]) -> None:
        if not rec.get("success"):
            artifacts["tools_failed"] += 1
            return

        artifacts["tools_ok"] += 1
        out = rec.get("output")
        if not isinstance(out, dict):
            return

        fpath = out.get("file_path")
        if not fpath or out.get("is_directory"):
            return

        real_path = os.path.realpath(str(fpath))
        if tool_name in ("file_read", "python_analyzer"):
            self._read_files.add(real_path)
            if real_path not in artifacts["files_read"]:
                artifacts["files_read"].append(real_path)
        elif tool_name in MUTATING_FILE_TOOLS:
            self._read_files.add(real_path)
            if real_path not in artifacts["files_modified"]:
                artifacts["files_modified"].append(real_path)

    async def _invoke_tool(self, tool_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        guard_err = self._preflight_guard(tool_name, payload)
        if guard_err is not None:
            return guard_err

        try:
            if hasattr(self.tool_manager, "execute_tool_result") and callable(self.tool_manager.execute_tool_result):
                res = await self.tool_manager.execute_tool_result(tool_name, payload)
            elif hasattr(self.tool_manager, "registry"):
                tool = self.tool_manager.registry.get_tool(tool_name)
                res = await tool.execute(**payload)
            else:
                return {
                    "tool": tool_name,
                    "arguments": payload,
                    "success": False,
                    "output": None,
                    "error": "ToolManager does not have a registry.",
                }

            res_dict = (
                res.model_dump()
                if hasattr(res, "model_dump")
                else (
                    res.dict()
                    if hasattr(res, "dict")
                    else {
                        "success": getattr(res, "success", False),
                        "output": getattr(res, "output", str(res)),
                        "error": getattr(res, "error", None),
                    }
                )
            )
            self._track_path_access(tool_name, res_dict)
            return {
                "tool": tool_name,
                "arguments": payload,
                "success": res_dict.get("success", False),
                "output": res_dict.get("output"),
                "error": res_dict.get("error"),
            }
        except Exception as e:
            return {
                "tool": tool_name,
                "arguments": payload,
                "success": False,
                "output": None,
                "error": str(e),
            }

    @staticmethod
    def _format_step_observations(step_idx: int, step_results: List[Dict[str, Any]]) -> str:
        lines: List[str] = []
        for rec in step_results:
            status = "OK" if rec.get("success") else "FAIL"
            detail = rec.get("output") if rec.get("success") else rec.get("error")
            lines.append(f"[Step {step_idx + 1} | {rec.get('tool')} | {status}]: {compact_value(detail, max_chars=700)}")
        return "\n".join(lines)

    async def generate_and_execute(
        self,
        task: Any,
        previous_results: str = "",
        context: str = "",
        max_tool_steps: int = 4,
    ) -> ActorOutputSchema:
        """Thinker-driven execution loop with safety guards, dedup loop break, and compact observations."""
        if not self.thinker:
            raise ValueError("Actor requires a Thinker instance to generate actions from a Task.")

        current_context = context
        if previous_results:
            current_context = f"{context}\n\nPrevious Results:\n{previous_results}".strip()

        all_results: List[Dict[str, Any]] = []
        thoughts: List[str] = []
        direct_responses: List[str] = []
        expected_outcomes: List[str] = []
        seen_calls: Set[tuple[str, str]] = set()
        artifacts: Dict[str, Any] = {
            "files_read": [],
            "files_modified": [],
            "tools_ok": 0,
            "tools_failed": 0,
        }

        task_type = str(getattr(task, "task_type", "")).lower()

        for step_idx in range(max(1, max_tool_steps)):
            try:
                task_execution = await self.thinker.solve_task(task, context=current_context)
            except (ValueError, RuntimeError) as fmt_err:
                if not isinstance(fmt_err, ValueError) and "json" not in str(fmt_err).lower():
                    raise
                current_context = (
                    f"{current_context}\n\n[Format Recovery]: {fmt_err}. "
                    "Respond ONLY with valid JSON containing `tools_to_call` or `direct_response`."
                ).strip()
                continue

            if getattr(task_execution, "thought_process", None):
                thoughts.append(task_execution.thought_process)
            if getattr(task_execution, "expected_outcome", None):
                expected_outcomes.append(str(task_execution.expected_outcome))
            if getattr(task_execution, "direct_response", None):
                direct_responses.append(task_execution.direct_response)

            tool_calls = getattr(task_execution, "tools_to_call", []) or []
            if not tool_calls:
                break

            step_sigs = [
                (tc.tool_name, compact_value(tc.arguments or {}, max_chars=240))
                for tc in tool_calls
            ]
            if step_idx > 0 and all(sig in seen_calls for sig in step_sigs):
                break
            seen_calls.update(step_sigs)

            step_results: List[Dict[str, Any]] = []
            step_tool_names: List[str] = []
            returned_direct = False

            for tc in tool_calls:
                t_name = tc.tool_name
                t_args = tc.arguments or {}
                step_tool_names.append(t_name)
                rec = await self._invoke_tool(t_name, t_args)
                self._record_side_effects(t_name, rec, artifacts)
                step_results.append(rec)
                all_results.append(rec)

                if rec.get("success") and self._is_return_direct_tool(t_name):
                    if rec.get("output") is not None:
                        direct_responses.append(str(rec["output"]))
                    returned_direct = True
                    break

            is_step_complete = bool(getattr(task_execution, "is_complete", False)) or returned_direct
            only_read_ops = bool(step_tool_names) and all(self._is_read_only_tool(t) for t in step_tool_names)
            needs_more = (not is_step_complete) or (only_read_ops and task_type in ("write", "update", "delete"))

            if returned_direct or not needs_more:
                break

            if step_idx + 1 < max_tool_steps:
                obs_digest = self._format_step_observations(step_idx, step_results)
                current_context = (
                    f"{current_context}\n\n"
                    f"[Tool Results]:\n{obs_digest}\n"
                    "Continue with the next tool call or set `is_complete: true` and summarize only verified outputs."
                ).strip()

        overall_success = bool(all_results or direct_responses) and all(r.get("success", False) for r in all_results)
        task_id = getattr(task, "task_id", "unknown")
        tool_label = (
            all_results[0]["tool"]
            if len(all_results) == 1
            else ("multiple_actions" if all_results else "direct_response")
        )
        result_payload: Dict[str, Any] = {
            "execution_results": all_results,
            "thought_process": "\n".join(thoughts),
        }
        if direct_responses:
            result_payload["direct_response"] = "\n".join(direct_responses)

        failed_errs = [r.get("error") for r in all_results if not r.get("success") and r.get("error")]
        err_ctx = "; ".join(str(e) for e in failed_errs[:2]) if not overall_success else None

        return ActorOutputSchema(
            task_id=task_id,
            tool_name=tool_label,
            success=overall_success,
            result=result_payload,
            error_context=err_ctx,
            expected_outcome=" | ".join(expected_outcomes) if expected_outcomes else None,
            artifacts=artifacts,
        )

    async def execute(self, task_input: Any, task_context: Optional[Any] = None) -> Any:
        """
        Executes either:
        1) An action-batch dictionary `{"actions": [{"tool": ..., "kwargs": ...}]}` in parallel (returning consolidated string), or
        2) A strict `ActorInputSchema` (returning `ActorOutputSchema`).
        """
        if isinstance(task_input, dict) and "actions" in task_input:
            actions = task_input.get("actions", [])
            if not actions:
                return "No actions provided."

            for act in actions:
                if act.get("tool") == "finish":
                    return "Task marked as finished by Planner."

            async def _run_single(act: Dict[str, Any]) -> str:
                t_name = act.get("tool", "")
                t_kwargs = act.get("kwargs", act.get("arguments", {})) or {}
                if hasattr(self.tool_manager, "execute_tool"):
                    return await self.tool_manager.execute_tool(t_name, t_kwargs)
                rec = await self._invoke_tool(t_name, t_kwargs)
                return str(rec.get("output") if rec.get("success") else rec.get("error"))

            outputs = await asyncio.gather(*[_run_single(a) for a in actions])
            return "\n---\n".join(str(o) for o in outputs)

        if isinstance(task_input, ActorInputSchema):
            rec = await self._invoke_tool(task_input.tool_name, task_input.payload)
            return ActorOutputSchema(
                task_id=task_input.task_id,
                tool_name=task_input.tool_name,
                success=rec.get("success", False),
                result={"execution_results": [rec]},
                error_context=rec.get("error"),
            )

        return await self.generate_and_execute(task_input, context=str(task_context or ""))
