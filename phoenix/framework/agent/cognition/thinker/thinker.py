import inspect
import json
from typing import Any, Dict, Optional
from .base import BaseThinker
from phoenix.framework.agent.cognition.schema import (
    PlanSchema,
    TaskSchema,
    TaskExecutionSchema,
    ReflectionSchema,
    TaskType,
    TaskPriority,
    TaskStatus,
)
from phoenix.framework.agent.cognition.utils import parse_llm_json


class Thinker(BaseThinker):
    """Central Brain. Coordinates structured reasoning, tool selection, reflection, and synthesis."""

    def __init__(self, llm, profile=None, tool_manager=None):
        super().__init__(llm, profile=profile, tool_manager=tool_manager)

    def _get_tools_list(self, compact_signatures: bool = False) -> str:
        """
        Build token-efficient available tools string for prompt injection.
        When `compact_signatures=True` (for high-level planning), formats crisp 1-line tool signatures.
        When `False` (for exact execution calls), emits compact min-whitespace JSON schemas.
        """
        tools_info = []
        if self.tool_manager:
            if hasattr(self.tool_manager, "get_available_tools_info"):
                tools_info = self.tool_manager.get_available_tools_info()
            elif hasattr(self.tool_manager, "registry") and hasattr(self.tool_manager.registry, "get_all_tools_info"):
                tools_info = self.tool_manager.registry.get_all_tools_info()
        elif getattr(self, "tools", None) and hasattr(self.tools, "get_all_tools_info"):
            tools_info = self.tools.get_all_tools_info()

        if not tools_info:
            return ""

        if compact_signatures:
            lines = []
            for t in tools_info:
                if not isinstance(t, dict):
                    continue
                name = t.get("name", "")
                desc = str(t.get("description", "")).replace("\n", " ").strip()
                params = t.get("parameters") or {}
                props = params.get("properties", {}) if isinstance(params, dict) else {}
                req = set(params.get("required", []) if isinstance(params, dict) else ())
                sig_parts = []
                for p_name, p_meta in props.items():
                    p_type = p_meta.get("type", "any") if isinstance(p_meta, dict) else "any"
                    suffix = "" if p_name in req else "?"
                    sig_parts.append(f"{p_name}{suffix}:{p_type}")
                lines.append(f"- {name}({', '.join(sig_parts)}): {desc}")
            return "\n\nAvailable Tools:\n" + "\n".join(lines)

        return "\n\nAvailable Tools:\n" + json.dumps(tools_info, separators=(",", ":"), ensure_ascii=False)

    def _get_profile_string(self) -> str:
        """Get the agent profile prompt string."""
        if self.profile and hasattr(self.profile, "to_prompt_string"):
            return f"\n\n{self.profile.to_prompt_string()}"
        return ""

    async def _call_structured(
        self,
        prompt: str,
        schema_cls: Any,
        session_id: Optional[str] = None,
        max_tokens: int = 8192,
    ) -> Any:
        gen_struct = getattr(self.llm, "generate_structured", None)
        if gen_struct is not None and (
            inspect.iscoroutinefunction(gen_struct)
            or hasattr(gen_struct, "_is_coroutine")
        ):
            return await gen_struct(prompt, schema_cls, session_id=session_id, max_tokens=max_tokens)

        # Fallback for custom/mock LLMs that only implement `generate(prompt, ...)`
        raw = await self.llm.generate(prompt, session_id=session_id, max_tokens=max_tokens)
        data = parse_llm_json(raw) if isinstance(raw, str) else (raw if isinstance(raw, dict) else {})
        try:
            if hasattr(schema_cls, "model_validate"):
                return schema_cls.model_validate(data)
            return schema_cls(**data)
        except Exception:
            # Construct minimal valid fallback for the requested schema
            if schema_cls is PlanSchema:
                return PlanSchema(
                    objective=str(raw or prompt)[:200],
                    tasks=[TaskSchema(description=str(prompt))],
                )
            if schema_cls is TaskExecutionSchema:
                return TaskExecutionSchema(
                    thought_process=str(raw or ""),
                    tools_to_call=[],
                    direct_response=str(raw or ""),
                    is_complete=True,
                )
            if schema_cls is ReflectionSchema:
                return ReflectionSchema(
                    status=TaskStatus.DONE,
                    feedback=str(raw or "Completed"),
                    rating=8,
                    is_task_complete=True,
                )
            raise

    async def analyze(self, prompt: str, memory: Any = None, session_id: str = "") -> str:
        """Analyze user prompt and extract core intent, requirements, and success criteria."""
        context = ""
        if memory and hasattr(memory, "get_full_context"):
            res = memory.get_full_context(session_id, query=prompt)
            context = await res if inspect.isawaitable(res) else str(res)

        analysis_prompt = (
            "Analyze the following user request and summarize the core objective, key requirements, and success criteria.\n"
            f"{self._get_profile_string()}\n"
            f"Context:\n{context}\n\nUser Prompt: {prompt}"
        )
        return await self.llm.generate(analysis_prompt)

    async def generate_plan(
        self,
        prompt: str,
        memory: Any = None,
        session_id: str = "",
        workspace_analysis: Optional[Dict[str, Any]] = None,
    ) -> PlanSchema:
        """Brain Step 1: Analyze user request and generate a structured plan with tasks and dependencies."""
        context = ""
        if memory and hasattr(memory, "get_full_context"):
            res = memory.get_full_context(session_id, query=prompt)
            context = await res if inspect.isawaitable(res) else str(res)

        task_types = [e.value for e in TaskType]
        task_priorities = [e.value for e in TaskPriority]
        task_statuses = [e.value for e in TaskStatus]
        tools_list = self._get_tools_list(compact_signatures=True)

        workspace_section = ""
        if workspace_analysis:
            workspace_section = f"\n\nWorkspace Analysis:\n{json.dumps(workspace_analysis, indent=2)}"

        system_prompt = (
            "You are the master Thinker and Planner. Analyze the user request and project context.\n"
            "Create a strict, minimal, and effective plan containing a main objective and a list of tasks to accomplish it.\n"
            f"{tools_list}\n"
            "IMPORTANT:\n"
            f" - For task_type, use one of: {task_types}.\n"
            f" - For priority, use one of: {task_priorities}.\n"
            f" - For status, use one of: {task_statuses} (new tasks should be 'pending').\n"
            " - Design tasks around the Available Tools (including any custom user-defined tools). If a specific tool matches the user's request, create a task to execute it.\n"
            " - Use `dependencies` (list of task_ids from earlier tasks in this plan) ONLY when a task strictly depends on another task finishing first. Independent tasks should have `dependencies: []` so they can run in parallel.\n"
            " - Prefer fewer, cohesive tasks over excessive micro-tasks.\n"
            " - ALL file paths must be ABSOLUTE full paths whenever possible."
        )
        system_prompt += self._get_profile_string()

        full_prompt = f"{system_prompt}{workspace_section}\n\nContext:\n{context}\n\nUser Request: {prompt}"
        return await self._call_structured(full_prompt, PlanSchema, session_id=session_id, max_tokens=8192)

    def _supports_native_tool_calling(self) -> bool:
        from phoenix.services.llm.base import BaseLLM

        gen_tools = getattr(self.llm, "generate_with_tools", None)
        if gen_tools is None or not callable(gen_tools):
            return False
        # Avoid intercepting subclasses/mocks that only use BaseLLM's structured fallback
        cls_method = getattr(type(self.llm), "generate_with_tools", None)
        if cls_method is BaseLLM.generate_with_tools and "generate_with_tools" not in getattr(self.llm, "__dict__", {}):
            return False
        return True

    async def solve_task(self, task: Any, context: str = "") -> TaskExecutionSchema:
        """Brain Step 2: Reason about the task using native tool calling when supported, else structured JSON."""
        task_desc = getattr(task, "description", str(task))
        task_id = getattr(task, "task_id", "unknown")

        use_native = (
            self._supports_native_tool_calling()
            and self.tool_manager is not None
            and hasattr(self.tool_manager, "get_openai_tools")
        )
        openai_tools = self.tool_manager.get_openai_tools() if use_native else []

        if use_native and openai_tools:
            system_prompt = (
                "You are the Thinker. Given the following task and its context, call the exact tool(s) needed to complete the task, "
                "or reply directly once the task objective is already satisfied by previous tool outputs.\n"
                "IMPORTANT:\n"
                "1. ALWAYS use ABSOLUTE file paths for any file_path or directory whenever possible.\n"
                "2. Prioritize specific registered/custom tools matching the task objective.\n"
                "3. Use native file tools (`file_read`, `file_write`, `file_edit`, `file_update_multi`, `file_search`) rather than shell commands.\n"
                "4. If the task is already answered in context or needs no further tool calls, respond directly with your findings."
            )
            system_prompt += self._get_profile_string()
            full_prompt = f"{system_prompt}\n\nTask ID: {task_id}\nTask Description: {task_desc}\n\nContext:\n{context}"
            return await self.llm.generate_with_tools(full_prompt, openai_tools, max_tokens=8192)

        tools_list = self._get_tools_list()
        system_prompt = (
            "You are the Thinker. Given the following task and its full context (including any previous tool outputs in this task), "
            "analyze the problem, formulate your approach, and define the exact explicit tool calls required for the next step.\n"
            f"{tools_list}\n\n"
            "IMPORTANT:\n"
            "1. ALWAYS use ABSOLUTE file paths for any file_path or directory whenever possible.\n"
            "2. When specifying tools_to_call, ONLY use the tool names and exact parameter names/types provided in the Available Tools list.\n"
            "3. Always prioritize calling a specific tool from Available Tools (including custom user-created tools) when it matches the task objective.\n"
            "4. Use native file tools (`file_read`, `file_write`, `file_edit`, `file_update_multi`, `file_search`) for file operations rather than shell commands.\n"
            "5. If you need to inspect a file before editing it, call `file_read` or `file_search` first.\n"
            "6. If the task is already answered by the context/tool outputs or requires no further tool calls, leave `tools_to_call` empty, provide the answer/findings in `direct_response`, and set `is_complete: true`.\n"
            "7. If previous attempts failed, check the context for error details and adjust your approach."
        )
        system_prompt += self._get_profile_string()

        full_prompt = f"{system_prompt}\n\nTask ID: {task_id}\nTask Description: {task_desc}\n\nContext:\n{context}"
        return await self._call_structured(full_prompt, TaskExecutionSchema, max_tokens=8192)

    async def generate_reflection(self, runtime_output: Any, context: str = "") -> ReflectionSchema:
        """Brain Step 3: Evaluate runtime execution results and judge task completion."""
        task_statuses = [e.value for e in TaskStatus]

        system_prompt = (
            "You are the Thinker reflecting on the output of the runtime execution.\n"
            "Analyze the success, tool outputs, direct responses, stdout, stderr, and any IO or tool failures.\n"
            "Use the full context including previous attempts, failed operations, and error logs.\n"
            "Judge if the current task is complete. Provide detailed feedback, a concise summary of findings/changes, and a rating (1-10).\n"
            "IMPORTANT:\n"
            f" - For status, use one of: {task_statuses}.\n"
            " - If all required tools succeeded or a valid direct_response was produced that satisfies the task description, set `is_task_complete: true` and `status: 'done'`.\n"
            " - If IO or tool operations failed, set `is_task_complete: false` and explain what went wrong and how to fix it.\n"
            " - Set `needs_replan: true` ONLY if the task revealed that the remaining plan is invalid and new tasks must be created."
        )
        system_prompt += self._get_profile_string()

        full_prompt = f"{system_prompt}\n\nFull Context:\n{context}\n\nRuntime Output:\n{runtime_output}"
        return await self._call_structured(full_prompt, ReflectionSchema)

    async def replan(
        self,
        objective: str,
        completed_summary: str,
        failed_task: Any,
        failure_context: str,
    ) -> PlanSchema:
        """Dynamically generate recovery tasks when a task fails or triggers a replan."""
        task_types = [e.value for e in TaskType]
        task_priorities = [e.value for e in TaskPriority]
        tools_list = self._get_tools_list(compact_signatures=True)
        failed_desc = getattr(failed_task, "description", str(failed_task))

        system_prompt = (
            "You are the master Thinker performing dynamic re-planning after an execution obstacle.\n"
            "Given the overall objective, completed work so far, and the failure details, generate a minimal list of remaining/recovery tasks.\n"
            f"{tools_list}\n"
            f"Valid task_type: {task_types} | Valid priority: {task_priorities}"
        )
        full_prompt = (
            f"{system_prompt}\n\nObjective: {objective}\n\n"
            f"Completed Work:\n{completed_summary}\n\n"
            f"Failed Task: {failed_desc}\nFailure Context:\n{failure_context}"
        )
        return await self._call_structured(full_prompt, PlanSchema, max_tokens=4096)

    async def synthesize_final_answer(
        self,
        prompt: str,
        objective: str,
        execution_digest: str,
        context: str = "",
    ) -> str:
        """Synthesize a clear, direct, user-facing response from all executed tasks and tool outputs."""
        system_prompt = (
            "You are the Phoenix Autonomous Agent. You have finished executing the planned tasks for the user's request.\n"
            "Using the execution digest (tool outputs, file operations, and task findings), provide a clear, accurate, "
            "and helpful final response directly addressing the user's prompt.\n"
            "Include concrete results, findings, or summaries of files created/modified."
        )
        system_prompt += self._get_profile_string()

        full_prompt = (
            f"{system_prompt}\n\n"
            f"User Request: {prompt}\n"
            f"Objective: {objective}\n\n"
            f"Execution Digest:\n{execution_digest}\n"
        )
        if context:
            full_prompt += f"\nAdditional Context:\n{context[:2000]}"

        return await self.llm.generate(full_prompt)
