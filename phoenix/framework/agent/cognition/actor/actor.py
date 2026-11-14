from __future__ import annotations
import asyncio
import json
from typing import Optional, Any, Dict, List, TYPE_CHECKING
from .base import BaseActor
from phoenix.framework.agent.cognition.schema import ActorInputSchema, ActorOutputSchema

if TYPE_CHECKING:
    from phoenix.framework.agent.cognition.schema import ActionSchema, SolutionSchema, ProblemSchema


READ_ONLY_TOOLS = {
    "file_read",
    "file_search",
    "python_analyzer",
    "web_search",
    "wikipedia_search",
    "arxiv_search",
    "web_scraper",
}

WRITE_KEYWORDS = ("write", "create", "edit", "update", "modify", "patch", "delete", "append", "implement", "fix")


class Actor(BaseActor):
    """Execution engine that binds Thinker outputs to Tools via ToolManager and Runtime."""

    def __init__(
        self,
        tool_manager: Any,
        thinker: Optional[Any] = None,
        runtime: Optional[Any] = None,
        reflector: Optional[Any] = None,
    ):
        super().__init__(tool_manager=tool_manager, thinker=thinker, reflector=reflector, runtime=runtime)

    async def _invoke_tool(self, tool_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
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

    def _needs_followup_action_step(self, task: Any, step_tools: List[str], is_complete: bool = False) -> bool:
        """
        Returns True if the current step only executed read/inspection tools
        and either the step was marked `is_complete=False` or the task requires modifying/writing files.
        """
        if not step_tools or not all(t in READ_ONLY_TOOLS for t in step_tools):
            return False
        if not is_complete:
            return True
        task_type = str(getattr(task, "task_type", "")).lower()
        if task_type in ("write", "update", "delete"):
            return True
        desc = str(getattr(task, "description", task)).lower()
        return any(kw in desc for kw in WRITE_KEYWORDS)

    async def generate_and_execute(
        self,
        task: Any,
        previous_results: str = "",
        context: str = "",
        max_tool_steps: int = 2,
    ) -> ActorOutputSchema:
        """Generate actions from Thinker then execute tools dynamically (with ReAct follow-up when inspecting before acting)."""
        if not self.thinker:
            raise ValueError("Actor requires a Thinker instance to generate actions from a Task.")

        current_context = context
        if previous_results:
            current_context = f"{context}\n\nPrevious Results:\n{previous_results}".strip()

        all_results: List[Dict[str, Any]] = []
        thoughts: List[str] = []
        direct_responses: List[str] = []
        overall_success = True

        for step_idx in range(max(1, max_tool_steps)):
            task_execution = await self.thinker.solve_task(task, context=current_context)
            if getattr(task_execution, "thought_process", None):
                thoughts.append(task_execution.thought_process)
            if getattr(task_execution, "direct_response", None):
                direct_responses.append(task_execution.direct_response)

            tool_calls = getattr(task_execution, "tools_to_call", []) or []
            if not tool_calls:
                break

            step_results: List[Dict[str, Any]] = []
            step_tool_names: List[str] = []

            for tool_call in tool_calls:
                t_name = tool_call.tool_name
                t_args = tool_call.arguments or {}
                step_tool_names.append(t_name)
                rec = await self._invoke_tool(t_name, t_args)
                step_results.append(rec)
                all_results.append(rec)
                if not rec.get("success", False):
                    overall_success = False

            if getattr(task_execution, "is_complete", False) or not overall_success:
                break

            if step_idx + 1 < max_tool_steps and self._needs_followup_action_step(task, step_tool_names):
                obs_str = json.dumps(step_results, indent=2, default=str)[:6000]
                current_context = (
                    f"{current_context}\n\n"
                    f"[Step {step_idx + 1} Inspection Results]:\n{obs_str}\n"
                    f"Now proceed with the required action/modification to complete the task."
                )
            else:
                break

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

        return ActorOutputSchema(
            task_id=task_id,
            tool_name=tool_label,
            success=overall_success,
            result=result_payload,
            error_context="Some tools failed" if not overall_success else None,
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
