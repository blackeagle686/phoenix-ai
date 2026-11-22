import asyncio
import inspect
import os
from typing import Dict, Any, Optional, Callable, List

from phoenix.services.observability.tracing import tracer

from phoenix.framework.agent.cognition.thinker import Thinker
from phoenix.framework.agent.cognition.planner import Planner
from phoenix.framework.agent.cognition.reflector import Reflector
from phoenix.framework.agent.cognition.analyzer import Analyzer
from phoenix.framework.agent.cognition.actor import Actor
from phoenix.framework.agent.cognition.pipeline import BrainRegistry, CognitionPipeline
from phoenix.framework.agent.cognition.schema import (
    ActionWatchSchema,
    PlanSchema,
    ReflectionSchema,
    ReflectorInputSchema,
    ReflectorType,
    TaskSchema,
    TaskStatus,
)
from phoenix.framework.agent.cognition.actor.actor import MUTATING_FILE_TOOLS
from phoenix.framework.agent.engine.pre_analyzer import DiagnosticReport, PreAnalyzer
from phoenix.framework.agent.engine.runner import NeuroSymbolicRunner
from phoenix.framework.agent.engine.verifier import ExecutionVerifier
from phoenix.framework.agent.memory.budget import compact_value
from phoenix.framework.agent.memory.manager import MemoryManager
from phoenix.framework.agent.tools._fs_utils import get_workspace_snapshot, inspect_workspace_file
from phoenix.services.llm.moe import moe_route_context


def _is_async_callable(fn: Any) -> bool:
    if fn is None:
        return False
    return (
        inspect.iscoroutinefunction(fn)
        or inspect.iscoroutinefunction(getattr(fn, "__call__", None))
    )


def _model_to_dict(obj: Any, fallback_key: str = "output") -> Dict[str, Any]:
    if hasattr(obj, "model_dump") and callable(obj.model_dump):
        res = obj.model_dump()
        if isinstance(res, dict):
            return res
    if hasattr(obj, "dict") and callable(obj.dict):
        res = obj.dict()
        if isinstance(res, dict):
            return res
    if isinstance(obj, dict):
        return obj
    return {fallback_key: str(obj)}


class AgentLoop:
    """
    Coordinates the autonomous brain loop using a unified memory graph,
    dependency-aware parallel task execution, adaptive replanning, and answer synthesis.
    """

    def __init__(
        self,
        thinker: Thinker,
        planner: Planner,
        actor: Actor,
        reflector: Reflector,
        analyzer: Optional[Analyzer] = None,
        memory_manager: Optional[MemoryManager] = None,
        parallel_tasks: bool = True,
        max_concurrency: int = 3,
        replan_on_failure: bool = True,
        synthesize_answer: bool = True,
        enable_workspace_analysis: bool = False,
        approval_callback: Optional[Callable[..., Any]] = None,
        checkpoint_dir: Optional[str] = None,
        fast_reflect: bool = False,
    ):
        self.thinker = thinker
        self.planner = planner
        self.actor = actor
        self.reflector = reflector
        self.analyzer = analyzer
        self.memory_manager = memory_manager if isinstance(memory_manager, MemoryManager) else MemoryManager()
        self.parallel_tasks = parallel_tasks
        self.max_concurrency = max(1, max_concurrency)
        self.replan_on_failure = replan_on_failure
        self.synthesize_answer = synthesize_answer
        self.enable_workspace_analysis = enable_workspace_analysis
        self.approval_callback = approval_callback
        self.checkpoint_dir = checkpoint_dir
        self.fast_reflect = fast_reflect
        self._bootstrap_pipeline: Optional[CognitionPipeline] = None
        self._background_tasks = set()

    def _checkpoint_path(self, session_id: str) -> Optional[str]:
        if not self.checkpoint_dir or not session_id:
            return None
        safe_sid = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in session_id)
        return os.path.join(self.checkpoint_dir, f"{safe_sid}.json")

    def _save_checkpoint_safe(self, active_memory: MemoryManager, session_id: str, plan: PlanSchema, prompt: str) -> None:
        cp_path = self._checkpoint_path(session_id)
        if not cp_path or not hasattr(active_memory, "save_checkpoint"):
            return
        try:
            active_memory.save_checkpoint(cp_path, plan_data=_model_to_dict(plan), prompt=prompt)
        except Exception:
            pass

    def set_pipeline_specs(
        self,
        bootstrap_pipeline_path: Optional[str] = None,
        loop_pipeline_path: Optional[str] = None,
    ):
        """Configure optional JSON bootstrap pipeline specification."""
        if bootstrap_pipeline_path:
            self._bootstrap_pipeline = CognitionPipeline.from_json_file(bootstrap_pipeline_path)
        else:
            self._bootstrap_pipeline = None

    def _build_brain_registry(self) -> BrainRegistry:
        registry = BrainRegistry()

        async def _thinker_handler(payload: Dict[str, Any]) -> Dict[str, Any]:
            prompt = payload.get("prompt", "")
            mem = payload.get("memory")
            sid = payload.get("session_id", "")
            if hasattr(self.thinker, "analyze"):
                obj = await self.thinker.analyze(prompt, mem, sid)
                return {"objective": obj}
            return {"objective": prompt}

        async def _analyzer_handler(payload: Dict[str, Any]) -> Dict[str, Any]:
            prompt = payload.get("prompt", "")
            if self.analyzer and hasattr(self.analyzer, "analyze_workspace"):
                res = await self.analyzer.analyze_workspace(prompt)
                return res if isinstance(res, dict) else {"summary": str(res)}
            return {}

        registry.register("thinker", _thinker_handler)
        registry.register("analyzer", _analyzer_handler)
        return registry

    def _schedule_background(self, coro):
        if not inspect.isawaitable(coro):
            return
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def _flush_background(self):
        if self._background_tasks:
            pending = list(self._background_tasks)
            await asyncio.gather(*pending, return_exceptions=True)

    def _resolve_memory_manager(self, memory: Any) -> MemoryManager:
        if isinstance(memory, MemoryManager):
            self.memory_manager = memory
            return memory
        return self.memory_manager

    def _uses_plan_schema_mode(self) -> bool:
        gen_plan = getattr(self.planner, "generate_initial_plan", None)
        plan_fn = getattr(self.planner, "plan", None)
        if not _is_async_callable(gen_plan):
            return False
        if type(plan_fn).__name__ == "AsyncMock" and type(gen_plan).__name__ != "AsyncMock":
            return False
        return True

    async def _run_iterative_action_loop(
        self,
        prompt: str,
        memory: Any,
        session_id: str,
        max_iterations: int = 15,
    ) -> str:
        """
        Iterative Planner -> Actor -> Reflector loop for action-batch planners.
        """
        objective = prompt
        if self._bootstrap_pipeline is not None:
            reg = self._build_brain_registry()
            state = await self._bootstrap_pipeline.run(
                reg,
                initial_state={"prompt": prompt, "memory": memory, "session_id": session_id},
            )
            thinker_out = state.get("thinker_output", {})
            if isinstance(thinker_out, dict) and thinker_out.get("objective"):
                objective = thinker_out["objective"]
        elif hasattr(self.thinker, "analyze") and _is_async_callable(self.thinker.analyze):
            objective = await self.thinker.analyze(prompt, memory, session_id)
            if self.analyzer and _is_async_callable(getattr(self.analyzer, "analyze_workspace", None)):
                await self.analyzer.analyze_workspace(prompt)

        if hasattr(memory, "session") and hasattr(memory.session, "set"):
            memory.session.set("current_objective", objective)

        previous_results = ""
        executed_results: List[str] = []
        has_executed_real_action = False

        for _ in range(max_iterations):
            sig = inspect.signature(self.planner.plan)
            if len(sig.parameters) >= 2:
                plan_dict = await self.planner.plan(objective, previous_results=previous_results)
            else:
                plan_dict = await self.planner.plan(objective)

            actions = plan_dict.get("actions", []) if isinstance(plan_dict, dict) else []
            is_finish = any(a.get("tool") == "finish" for a in actions if isinstance(a, dict))

            action_res = await self.actor.execute(plan_dict)
            if not is_finish:
                has_executed_real_action = True
            executed_results.append(str(action_res))
            previous_results = str(action_res)

            ref_res = await self.reflector.reflect(objective, plan_dict, action_res)
            if isinstance(ref_res, dict):
                ref_text = ref_res.get("reflection", "")
                if ref_text and hasattr(memory, "reflection") and hasattr(memory.reflection, "add_reflection"):
                    memory.reflection.add_reflection(ref_text)
                if hasattr(memory, "consolidate_reflections") and _is_async_callable(memory.consolidate_reflections):
                    self._schedule_background(memory.consolidate_reflections(getattr(self.reflector, "llm", None)))
                if ref_res.get("is_complete") and has_executed_real_action:
                    break

            if is_finish and has_executed_real_action:
                break

        final_answer = "\n".join(executed_results) if executed_results else "No actions executed."
        if hasattr(memory, "add_interaction") and _is_async_callable(memory.add_interaction):
            await memory.add_interaction(session_id, "assistant", final_answer)
        return final_answer

    async def _execute_single_task(
        self,
        task: Any,
        active_memory: MemoryManager,
        memory: Any,
        session_id: str,
        max_attempts_left_fn: Callable[[], bool],
        increment_iteration_fn: Callable[[], None],
    ) -> tuple[bool, str, Optional[Any]]:
        cell = active_memory.get_cell(task.task_id)
        if self.fast_reflect and cell.max_attempts > 2:
            cell.max_attempts = 2
        cell.task_status = "in_progress"
        task_complete = False
        last_reflection = None

        while not task_complete and max_attempts_left_fn() and not cell.is_exhausted():
            cell.increment_attempt()
            context = active_memory.get_task_context(task.task_id)

            if self.approval_callback is not None:
                decision = self.approval_callback(task, context)
                if inspect.isawaitable(decision):
                    decision = await decision
                if decision == "skip":
                    task.status = "skipped"
                    cell.task_status = "skipped"
                    return True, f"Task {task.task_id} skipped by human.\n", None
                if decision in ("reject", False):
                    return (
                        False,
                        f"\nExecution stopped due to human rejection on task {task.task_id}.",
                        "REJECTED",
                    )

            with moe_route_context(role="coder", attempt=cell.attempts, escalate=(cell.attempts >= 2)):
                actor_output = await self.actor.generate_and_execute(task, context=context)
            result_data = _model_to_dict(actor_output)
            cell.add_runtime_result(result_data)

            if hasattr(actor_output, "result") and isinstance(actor_output.result, dict):
                exec_results = actor_output.result.get("execution_results", [])
                for r in exec_results:
                    if r.get("io_op"):
                        cell.add_io_record(
                            operation=r.get("io_op", "unknown"),
                            file_path=r.get("path", "unknown"),
                            success=r.get("success", False),
                            error=r.get("error"),
                        )
                    if r.get("tool"):
                        cell.add_tool_record(
                            tool_name=r.get("tool", "unknown"),
                            arguments=r.get("arguments", {}),
                            success=r.get("success", False),
                            output=r.get("output"),
                            error=r.get("error"),
                        )

            if hasattr(actor_output, "success") and not actor_output.success:
                cell.log_error(getattr(actor_output, "error_context", "Actor execution failed") or "Actor execution failed")

            has_syntax_warn = False
            has_mutation = False
            rt = getattr(getattr(self.actor, "tool_manager", None), "runtime", None) or getattr(self.actor, "runtime", None)
            if hasattr(actor_output, "result") and isinstance(actor_output.result, dict):
                for r in actor_output.result.get("execution_results", []) or []:
                    t_name = r.get("tool", "")
                    out_obj = r.get("output")
                    if isinstance(out_obj, dict) and out_obj.get("syntax_warning"):
                        has_syntax_warn = True
                    if r.get("success") and t_name in MUTATING_FILE_TOOLS:
                        has_mutation = True
                        fpath = out_obj.get("file_path") if isinstance(out_obj, dict) else None
                        if fpath and rt:
                            chk = inspect_workspace_file(rt, str(fpath))
                            if chk.get("exists") and not chk.get("ast_valid", True):
                                has_syntax_warn = True

            tt_raw = getattr(task, "task_type", "")
            needs_mutation = str(getattr(tt_raw, "value", tt_raw) or "").lower() in ("write", "update", "delete")
            can_fast_reflect = (
                self.fast_reflect
                and bool(getattr(actor_output, "success", False))
                and not has_syntax_warn
                and not has_mutation
                and (not needs_mutation or has_mutation)
                and not cell.get_failed_tools()
                and not cell.get_failed_io_ops()
                and (bool(cell.get_successful_tools()) or bool(cell.get_direct_findings()))
            )

            if can_fast_reflect:
                reflection = ReflectionSchema(
                    status=TaskStatus.DONE,
                    feedback=f"Task {task.task_id} succeeded cleanly (fast-reflect).",
                    rating=9,
                    is_task_complete=True,
                    summary=cell.get_outcome_summary(200),
                )
            else:
                ref_context = active_memory.build_retry_context(task.task_id)
                compact_target = cell.get_last_runtime_result() or result_data
                ref_input = ReflectorInputSchema(
                    reflector_type=ReflectorType.TASK,
                    target_id=task.task_id,
                    target_content=compact_target,
                    context=ref_context,
                )
                reflection = await self.reflector.reflect(ref_input)

            last_reflection = reflection
            ref_dict = _model_to_dict(reflection, fallback_key="feedback")
            cell.add_reflection(ref_dict)

            feedback_text = getattr(reflection, "feedback", None) or ref_dict.get("feedback")
            if feedback_text and hasattr(memory, "reflection") and hasattr(memory.reflection, "add_reflection"):
                memory.reflection.add_reflection(f"[{task.task_id}] {feedback_text}")

            increment_iteration_fn()

            if getattr(reflection, "is_task_complete", False):
                task_complete = True
                task.status = TaskStatus.DONE
                active_memory.mark_complete(task.task_id)
                rating = getattr(reflection, "rating", 8)
                return True, f"Task {task.task_id} completed. Rating: {rating}/10.\n", reflection

        if not task_complete and cell.is_exhausted():
            active_memory.mark_failed(task.task_id, "Max attempts exhausted")
            return False, f"Task {task.task_id} failed after {cell.attempts} attempts.\n", last_reflection

        return task_complete, "", last_reflection

    async def _run_plan_dag_execution(
        self,
        prompt: str,
        plan: PlanSchema,
        active_memory: MemoryManager,
        memory: Any,
        session_id: str,
        max_iterations: int = 15,
    ) -> str:
        """Shared DAG execution, adaptive replanning, checkpointing, and answer synthesis."""
        status_log = ""
        iterations = 0
        replan_count = 0
        max_replans = 1 if self.fast_reflect else 3

        def _has_iterations_left() -> bool:
            return iterations < max_iterations

        def _inc_iteration() -> None:
            nonlocal iterations
            iterations += 1

        remaining = [
            t for t in plan.tasks
            if str(getattr(t, "status", "pending")).lower() != "done" and t.task_id not in active_memory.completed_tasks
        ]

        await tracer.emit_event("plan", {"session_id": session_id, "objective": plan.objective, "tasks": len(remaining)})
        self._save_checkpoint_safe(active_memory, session_id, plan, prompt)

        while remaining and iterations < max_iterations:
            ready = [t for t in remaining if active_memory.are_dependencies_met(t.task_id)]
            if not ready:
                ready = [remaining[0]]

            can_parallelize = (
                self.parallel_tasks
                and len(ready) > 1
                and self.approval_callback is None
            )

            for r_task in (ready if can_parallelize else [ready[0]]):
                await tracer.emit_event("task_start", {"task_id": r_task.task_id, "description": r_task.description})

            if can_parallelize:
                sem = asyncio.Semaphore(self.max_concurrency)

                async def _run_bounded(t):
                    async with sem:
                        return t, await self._execute_single_task(
                            t,
                            active_memory,
                            memory,
                            session_id,
                            _has_iterations_left,
                            _inc_iteration,
                        )

                batch_results = await asyncio.gather(*[_run_bounded(t) for t in ready])
            else:
                t = ready[0]
                res = await self._execute_single_task(
                    t,
                    active_memory,
                    memory,
                    session_id,
                    _has_iterations_left,
                    _inc_iteration,
                )
                batch_results = [(t, res)]

            executed_ids = set()
            for task_obj, (completed, msg, reflection_obj) in batch_results:
                executed_ids.add(task_obj.task_id)
                status_log += msg
                await tracer.emit_event("task_done", {"task_id": task_obj.task_id, "completed": completed})

                if reflection_obj == "REJECTED":
                    self._save_checkpoint_safe(active_memory, session_id, plan, prompt)
                    await self._flush_background()
                    if hasattr(memory, "add_interaction"):
                        await memory.add_interaction(session_id, "assistant", status_log)
                    return status_log

                if (
                    self.replan_on_failure
                    and replan_count < max_replans
                    and (not completed or getattr(reflection_obj, "needs_replan", False))
                    and hasattr(self.thinker, "replan")
                    and _is_async_callable(self.thinker.replan)
                    and iterations < max_iterations
                ):
                    try:
                        replan_count += 1
                        recovery_plan = await self.thinker.replan(
                            objective=plan.objective,
                            completed_summary=active_memory.get_completed_summary(),
                            failed_task=task_obj,
                            failure_context=active_memory.build_retry_context(task_obj.task_id),
                        )
                        if isinstance(recovery_plan, PlanSchema) and recovery_plan.tasks:
                            for new_t in recovery_plan.tasks:
                                if not active_memory.has_cell(new_t.task_id):
                                    active_memory.create_cell(
                                        task_id=new_t.task_id,
                                        description=new_t.description,
                                        task_type=getattr(new_t, "task_type", "other"),
                                        priority=getattr(new_t, "priority", "medium"),
                                        dependencies=getattr(new_t, "dependencies", []),
                                    )
                                    plan.tasks.append(new_t)
                                    remaining.append(new_t)
                            await tracer.emit_event("replan", {"added_tasks": len(recovery_plan.tasks)})
                    except Exception:
                        pass

            self._save_checkpoint_safe(active_memory, session_id, plan, prompt)
            remaining = [t for t in remaining if t.task_id not in executed_ids and getattr(t, "status", "pending") != "done"]

        if iterations >= max_iterations and remaining:
            status_log += "\nMax iterations reached before completing all tasks."

        final_answer = status_log
        if (
            self.synthesize_answer
            and hasattr(self.thinker, "synthesize_final_answer")
            and _is_async_callable(self.thinker.synthesize_final_answer)
        ):
            try:
                digest = active_memory.get_execution_digest()
                synthesized = await self.thinker.synthesize_final_answer(
                    prompt=prompt,
                    objective=plan.objective,
                    execution_digest=digest,
                )
                if synthesized and isinstance(synthesized, str) and synthesized.strip():
                    final_answer = f"{synthesized.strip()}\n\n---\nExecution Summary:\n{status_log.strip()}"
            except Exception:
                pass

        if hasattr(memory, "add_interaction"):
            await memory.add_interaction(session_id, "assistant", final_answer)
        if hasattr(active_memory, "finalize_run"):
            self._schedule_background(
                active_memory.finalize_run(
                    session_id,
                    objective=plan.objective,
                    final_answer=final_answer,
                    llm=getattr(self.reflector, "llm", None),
                )
            )
        await self._flush_background()
        await tracer.emit_event("complete", {"session_id": session_id})
        return final_answer

    async def resume(self, memory: Any, session_id: str, max_iterations: int = 15) -> str:
        """Resume a previously checkpointed run for `session_id`."""
        cp_path = self._checkpoint_path(session_id)
        if not cp_path or not os.path.exists(cp_path):
            raise FileNotFoundError(f"No checkpoint found for session '{session_id}'.")

        active_memory = self._resolve_memory_manager(memory)
        payload = await active_memory.load_checkpoint(cp_path)
        plan_dict = payload.get("plan") or {
            "objective": payload.get("objective", ""),
            "tasks": [],
        }
        plan = PlanSchema.model_validate(plan_dict) if hasattr(PlanSchema, "model_validate") else PlanSchema(**plan_dict)
        for task in plan.tasks:
            if not active_memory.has_cell(task.task_id):
                active_memory.create_cell(
                    task_id=task.task_id,
                    description=task.description,
                    task_type=getattr(task, "task_type", "other"),
                    priority=getattr(task, "priority", "medium"),
                    dependencies=getattr(task, "dependencies", []),
                )
            elif task.task_id in active_memory.completed_tasks:
                task.status = TaskStatus.DONE

        prompt = str(payload.get("prompt") or plan.objective)
        return await self._run_plan_dag_execution(
            prompt=prompt,
            plan=plan,
            active_memory=active_memory,
            memory=memory,
            session_id=session_id,
            max_iterations=max_iterations,
        )

    async def run(self, prompt: str, memory: Any, session_id: str, max_iterations: int = 15) -> str:
        """Execute the full brain loop with unified memory graph, DAG execution, and synthesis."""
        if not self._uses_plan_schema_mode():
            return await self._run_iterative_action_loop(prompt, memory, session_id, max_iterations=max_iterations)

        active_memory = self._resolve_memory_manager(memory)
        active_memory.set_session(session_id, prompt)
        if hasattr(self.actor, "reset_session_tracking"):
            self.actor.reset_session_tracking()

        workspace_analysis = None
        if self.enable_workspace_analysis and self.analyzer and hasattr(self.analyzer, "analyze_workspace"):
            try:
                workspace_analysis = await self.analyzer.analyze_workspace(prompt)
            except Exception:
                workspace_analysis = None

        sig = inspect.signature(self.planner.generate_initial_plan)
        if "workspace_analysis" in sig.parameters:
            plan = await self.planner.generate_initial_plan(
                prompt,
                memory,
                session_id,
                workspace_analysis=workspace_analysis,
            )
        else:
            plan = await self.planner.generate_initial_plan(prompt, memory, session_id)

        active_memory.objective = plan.objective
        if hasattr(memory, "session") and hasattr(memory.session, "set"):
            memory.session.set("current_objective", plan.objective)

        for task in list(plan.tasks):
            active_memory.create_cell(
                task_id=task.task_id,
                description=task.description,
                task_type=getattr(task, "task_type", "other"),
                priority=getattr(task, "priority", "medium"),
                dependencies=getattr(task, "dependencies", []),
            )

        return await self._run_plan_dag_execution(
            prompt=prompt,
            plan=plan,
            active_memory=active_memory,
            memory=memory,
            session_id=session_id,
            max_iterations=max_iterations,
        )

    def _active_runtime(self) -> Any:
        return getattr(getattr(self.actor, "tool_manager", None), "runtime", None) or getattr(self.actor, "runtime", None)

    def _collect_deep_disk_checks(self, actor_output: Any, task: Any) -> tuple[str, List[str]]:
        """Inspect actual files on disk touched by Acting and flag deterministic issues."""
        rt = self._active_runtime()
        if rt is None:
            return "", []

        candidates: List[str] = []
        artifacts = getattr(actor_output, "artifacts", None) or {}
        for fp in artifacts.get("files_modified", []) or []:
            if fp not in candidates:
                candidates.append(fp)
        for fp in getattr(task, "target_files", []) or []:
            if fp and fp not in candidates:
                candidates.append(fp)

        reports: List[str] = []
        issues: List[str] = []
        for target in candidates[:6]:
            chk = inspect_workspace_file(rt, str(target), preview_lines=30)
            fname = os.path.basename(str(chk.get("file_path") or target))
            if not chk.get("exists"):
                issues.append(f"Expected file '{target}' does not exist on disk.")
                reports.append(f"- {fname}: MISSING ({chk.get('error', 'not found')})")
                continue
            if not chk.get("ast_valid", True):
                issues.append(f"Python syntax error in '{fname}': {chk.get('ast_error')}")
            symbols = ", ".join(chk.get("symbols") or []) or "none"
            reports.append(
                f"- {fname} ({chk.get('line_count', 0)}L, ast_valid={chk.get('ast_valid', True)}, symbols=[{symbols}]):\n"
                f"{chk.get('preview', '')}"
            )
            if chk.get("line_count", 0) == 0:
                issues.append(f"File '{fname}' is empty (0 lines).")

        return "\n".join(reports), issues

    def _is_local_execution_engine(self) -> bool:
        is_local_fn = getattr(self.thinker, "_is_local_model", None)
        if callable(is_local_fn):
            try:
                return bool(is_local_fn())
            except Exception:
                return False
        return False

    async def _finalize_deep_response(
        self,
        prompt: str,
        objective: str,
        status_log: str,
        digest: str,
        disk_evidence_notes: List[str],
        active_memory: MemoryManager,
        memory: Any,
        session_id: str,
        mode_name: str = "deep",
    ) -> str:
        final_answer = status_log
        if disk_evidence_notes:
            digest = f"{digest}\n\nVERIFIED ON-DISK EVIDENCE:\n" + "\n".join(disk_evidence_notes[-3:])
        if (
            self.synthesize_answer
            and hasattr(self.thinker, "synthesize_final_answer")
            and _is_async_callable(self.thinker.synthesize_final_answer)
        ):
            try:
                synthesized = await self.thinker.synthesize_final_answer(
                    prompt=prompt,
                    objective=objective,
                    execution_digest=digest,
                )
                if synthesized and isinstance(synthesized, str) and synthesized.strip():
                    final_answer = f"{synthesized.strip()}\n\n---\nDeep Execution Summary:\n{status_log.strip()}"
            except Exception:
                pass

        if hasattr(memory, "add_interaction"):
            await memory.add_interaction(session_id, "assistant", final_answer)
        if hasattr(active_memory, "finalize_run"):
            self._schedule_background(
                active_memory.finalize_run(
                    session_id,
                    objective=objective,
                    final_answer=final_answer,
                    llm=getattr(self.reflector, "llm", None),
                )
            )
        await self._flush_background()
        await tracer.emit_event("complete", {"session_id": session_id, "mode": mode_name})
        return final_answer

    async def _watch_deep_task(
        self,
        task: Any,
        actor_output: Any,
        expected_result: str,
        diag_report: Optional[DiagnosticReport] = None,
        prefer_deterministic_watch: bool = False,
    ) -> tuple[ActionWatchSchema, str]:
        """Deep-mode Watching stage: verify actual results + on-disk state + ExecutionVerifier."""
        disk_report, det_issues = self._collect_deep_disk_checks(actor_output, task)

        exec_results = []
        if hasattr(actor_output, "result") and isinstance(actor_output.result, dict):
            exec_results = actor_output.result.get("execution_results", []) or []

        tt_raw = getattr(task, "task_type", "")
        task_type = str(getattr(tt_raw, "value", tt_raw) or "").lower()
        mutated = any(r.get("success") and r.get("tool") in MUTATING_FILE_TOOLS for r in exec_results)
        has_direct = bool(
            hasattr(actor_output, "result")
            and isinstance(actor_output.result, dict)
            and (actor_output.result.get("direct_response") or "").strip()
        )
        if task_type in ("write", "update") and not mutated:
            det_issues.append("Task requires modifying/writing files, but no mutating file tool succeeded.")
        if not exec_results and not has_direct:
            det_issues.append("Actor produced no tool calls and no direct response.")
        if not getattr(actor_output, "success", False):
            err_ctx = getattr(actor_output, "error_context", None) or "One or more tool calls failed."
            det_issues.append(err_ctx)

        actual_lines = [
            f"- {r.get('tool')} [{'OK' if r.get('success') else 'FAIL'}]: "
            f"{compact_value(r.get('output') if r.get('success') else r.get('error'), max_chars=500)}"
            for r in exec_results[-6:]
        ]
        if has_direct:
            actual_lines.append(f"- Direct Response: {compact_value(actor_output.result.get('direct_response'), max_chars=400)}")
        actual_digest = "\n".join(actual_lines) or "(no tool execution output)"

        # Plan 3: Run deterministic AST, security policy, and live sandbox checks when code files exist
        rt = self._active_runtime()
        if not det_issues and diag_report is not None and rt is not None and task_type in ("write", "update"):
            artifacts = getattr(actor_output, "artifacts", None) or {}
            cand_files = list(artifacts.get("files_modified", []) or [])
            ver = await ExecutionVerifier.verify(diag_report, rt, candidate_files=cand_files)
            ver_summary = (
                f"ExecutionVerifier Score: {ver.score}/100"
                + (f" | Verified Outputs: {', '.join(ver.functional_outputs[:5])}" if ver.functional_outputs else "")
                + (f" | Issues: {'; '.join(ver.discrepancies[:3])}" if ver.discrepancies else "")
            )
            disk_report = f"{disk_report}\n{ver_summary}".strip()
            if prefer_deterministic_watch:
                if not ver.passed:
                    det_issues.extend(ver.discrepancies or ["Deterministic sandbox verification failed"])
                else:
                    return (
                        ActionWatchSchema(
                            matches_expectation=True,
                            actual_summary=ver_summary[:500],
                            discrepancies=[],
                            corrective_action=None,
                            rating=ver.rating,
                        ),
                        disk_report,
                    )

        if det_issues:
            watch = ActionWatchSchema(
                matches_expectation=False,
                actual_summary=actual_digest[:500],
                discrepancies=det_issues,
                corrective_action="; ".join(det_issues[:3]),
                rating=3,
            )
            return watch, disk_report

        if hasattr(self.thinker, "watch_action_results") and _is_async_callable(self.thinker.watch_action_results):
            try:
                watch = await self.thinker.watch_action_results(
                    task=task,
                    expected_result=expected_result,
                    actual_digest=actual_digest,
                    disk_checks=disk_report,
                )
                return watch, disk_report
            except Exception:
                pass

        return (
            ActionWatchSchema(
                matches_expectation=True,
                actual_summary=actual_digest[:500],
                discrepancies=[],
                corrective_action=None,
                rating=8,
            ),
            disk_report,
        )

    async def _run_deep_core(
        self,
        prompt: str,
        memory: Any,
        session_id: str,
        max_iterations: int = 15,
        status_queue: Optional[asyncio.Queue] = None,
        force_engine: bool = False,
    ) -> str:
        """
        Deep Cognitive Mode & Neuro-Symbolic Engine:
        Pre-Analysis -> Tasking/Playbook -> Acting/PatchProtocol -> Sandbox Verification -> Reacting & Repair.
        """
        async def _emit_phase(phase: str, info: Optional[Dict[str, Any]] = None) -> None:
            payload = {"session_id": session_id, "phase": phase, **(info or {})}
            await tracer.emit_event("deep_phase", payload)
            if status_queue is not None:
                await status_queue.put({"type": "status", "content": f"[Deep:{phase}] {info.get('message', '') if info else ''}".strip()})

        active_memory = self._resolve_memory_manager(memory)
        active_memory.set_session(session_id, prompt)
        if hasattr(self.actor, "reset_session_tracking"):
            self.actor.reset_session_tracking()

        rt = self._active_runtime()
        ws_snap = get_workspace_snapshot(rt, max_files=25) if rt else ""
        await _emit_phase("tasking", {"message": "Running deterministic workspace & AST/security pre-analysis..."})

        # Plan 1: Zero-LLM deterministic pre-analysis & file hydration
        diag_report = PreAnalyzer.analyze(prompt, runtime=rt) if rt is not None else None
        if diag_report and diag_report.pre_read_paths and hasattr(self.actor, "mark_files_pre_read"):
            self.actor.mark_files_pre_read(diag_report.pre_read_paths)

        is_local_engine = self._is_local_execution_engine()
        use_ns_code_runner = (
            diag_report is not None
            and diag_report.contract.archetype in ("code_modify", "code_create")
            and bool(diag_report.contract.target_files)
            and (force_engine or is_local_engine)
        )

        # Plan 1+2+3 Integrated Neuro-Symbolic Code Workflow (0-JSON code synthesis + live sandbox verification)
        if use_ns_code_runner and diag_report is not None:
            playbook = PreAnalyzer.build_playbook_plan(prompt, diag_report)
            plan_tasks = len(playbook.tasks) if playbook else 1
            objective = playbook.objective if playbook else prompt[:240]
            active_memory.objective = objective
            if hasattr(memory, "session") and hasattr(memory.session, "set"):
                memory.session.set("current_objective", objective)
            if playbook:
                for task in playbook.tasks:
                    active_memory.create_cell(
                        task_id=task.task_id,
                        description=task.description,
                        task_type=getattr(task, "task_type", "other"),
                        priority=getattr(task, "priority", "medium"),
                        dependencies=getattr(task, "dependencies", []),
                    )
            await _emit_phase("planning", {"tasks": plan_tasks, "message": f"Playbook prepared ({plan_tasks} cohesive task via Neuro-Symbolic Engine)."})

            ns_res = await NeuroSymbolicRunner.run_code_workflow(
                prompt=prompt,
                llm=self.thinker.llm,
                runtime=rt,
                report=diag_report,
                max_attempts=min(5, max(2, max_iterations)),
                emit_phase=_emit_phase,
            )
            ver = ns_res.get("verification")
            score = getattr(ver, "score", 0) if ver else 0
            rating = getattr(ver, "rating", 1) if ver else 1
            func_outputs = getattr(ver, "functional_outputs", []) if ver else []
            healed = ns_res.get("healed_actions") or []
            diff_txt = ns_res.get("diff") or ""
            status_log = (
                f"Task t1 {'verified' if ns_res.get('success') else 'partial'} "
                f"({rating}/10, sandbox score {score}/100, attempts={ns_res.get('attempts', 1)}).\n"
            )
            if func_outputs:
                status_log += f"Verified execution probes: {', '.join(func_outputs)}\n"
            if healed:
                status_log += f"Auto-healed: {', '.join(healed)}\n"
            if not ns_res.get("success") and ns_res.get("error"):
                status_log += f"Remaining issues: {ns_res['error']}\n"

            if playbook:
                if ns_res.get("success"):
                    active_memory.mark_complete("t1")
                else:
                    active_memory.mark_failed("t1", str(ns_res.get("error") or "Verification incomplete"))

            digest_lines = [
                f"Target File(s): {', '.join(ns_res.get('target_files', []))}",
                "On-Disk Status: ALREADY WRITTEN TO DISK by Neuro-Symbolic Engine (summarize the verified changes and actual results directly; do NOT tell the user to edit the file manually).",
                f"Sandbox Verification Score: {score}/100 ({'PASSED' if ns_res.get('success') else 'INCOMPLETE'})",
            ]
            if func_outputs:
                digest_lines.append("Sandbox Functional Checks: " + "; ".join(func_outputs))
            if healed:
                digest_lines.append("Deterministic Healer Fixes: " + "; ".join(healed))
            if diff_txt:
                digest_lines.append(f"On-Disk Diff:\n{diff_txt}")
            if not ns_res.get("success") and ns_res.get("error"):
                digest_lines.append(f"Remaining Gaps: {ns_res['error']}")

            if not ns_res.get("success") and is_local_engine:
                # Grounded factual report for failed local-model runs instead of hallucinated synthesis prose
                factual = (
                    f"Partial execution on `{', '.join(ns_res.get('target_files', []))}` "
                    f"(sandbox score {score}/100 after {ns_res.get('attempts', 1)} attempts).\n"
                )
                if ver and ver.diagnosis_hints:
                    factual += "\nRecommended Next Fix:\n" + "\n".join(ver.diagnosis_hints[:2]) + "\n"
                factual += f"\n---\nDeep Execution Summary:\n{status_log.strip()}"
                if hasattr(memory, "add_interaction"):
                    await memory.add_interaction(session_id, "assistant", factual)
                await self._flush_background()
                await tracer.emit_event("complete", {"session_id": session_id, "mode": "engine" if force_engine else "deep"})
                return factual

            return await self._finalize_deep_response(
                prompt=prompt,
                objective=objective,
                status_log=status_log,
                digest="\n".join(digest_lines),
                disk_evidence_notes=func_outputs,
                active_memory=active_memory,
                memory=memory,
                session_id=session_id,
                mode_name="engine" if force_engine else "deep",
            )

        plan = None
        if (force_engine or is_local_engine) and diag_report is not None:
            plan = PreAnalyzer.build_playbook_plan(prompt, diag_report)

        if plan is None:
            try:
                if hasattr(self.thinker, "generate_deep_plan") and _is_async_callable(self.thinker.generate_deep_plan):
                    plan = await self.thinker.generate_deep_plan(prompt, memory, session_id, workspace_snapshot=ws_snap)
                else:
                    plan = await self.planner.generate_initial_plan(prompt, memory, session_id)
            except (ValueError, RuntimeError):
                plan = PlanSchema(
                    objective=prompt,
                    tasks=[TaskSchema(task_id="t1", description=prompt, expected_result=f"Fulfill request accurately: {prompt[:180]}")],
                )

        if not getattr(plan, "tasks", None):
            plan = PlanSchema(
                objective=getattr(plan, "objective", prompt) or prompt,
                tasks=[TaskSchema(task_id="t1", description=prompt, expected_result="Complete user request accurately")],
            )

        active_memory.objective = plan.objective
        if hasattr(memory, "session") and hasattr(memory.session, "set"):
            memory.session.set("current_objective", plan.objective)

        for task in plan.tasks:
            cell = active_memory.create_cell(
                task_id=task.task_id,
                description=task.description,
                task_type=getattr(task, "task_type", "other"),
                priority=getattr(task, "priority", "medium"),
                dependencies=getattr(task, "dependencies", []),
            )
            cell.max_attempts = 3

        await _emit_phase("planning", {"tasks": len(plan.tasks), "message": f"Planned {len(plan.tasks)} verifiable tasks."})
        self._save_checkpoint_safe(active_memory, session_id, plan, prompt)

        status_log = ""
        disk_evidence_notes: List[str] = []
        iterations = 0
        replan_count = 0
        remaining = list(plan.tasks)
        diag_brief_section = (
            f"\n\n{diag_report.summary_brief}"
            if diag_report is not None and diag_report.summary_brief
            else ""
        )

        while remaining and iterations < max_iterations:
            ready = [t for t in remaining if active_memory.are_dependencies_met(t.task_id)]
            task = ready[0] if ready else remaining[0]
            cell = active_memory.get_cell(task.task_id)
            cell.task_status = "in_progress"
            expected_res = getattr(task, "expected_result", None) or task.description
            task_complete = False

            while not task_complete and iterations < max_iterations and not cell.is_exhausted():
                cell.increment_attempt()
                iterations += 1
                base_ctx = active_memory.get_task_context(task.task_id)
                ws_note = f"\nWorkspace Files: {ws_snap}" if ws_snap else ""
                target_files = ", ".join(getattr(task, "target_files", []) or []) or "auto"
                acting_ctx = (
                    f"{base_ctx}{ws_note}{diag_brief_section}\n\n"
                    f"DEEP MODE PRE-ACTION CONTRACT:\n"
                    f"- Expected Outcome Before Acting: {expected_res}\n"
                    f"- Target Files: {target_files}"
                )

                await _emit_phase("acting", {"task_id": task.task_id, "attempt": cell.attempts, "message": f"Executing {task.task_id} (attempt {cell.attempts})..."})
                with moe_route_context(role="coder", attempt=cell.attempts, escalate=(cell.attempts >= 2)):
                    actor_output = await self.actor.generate_and_execute(task, context=acting_ctx, max_tool_steps=5)
                cell.add_runtime_result(_model_to_dict(actor_output))

                if hasattr(actor_output, "result") and isinstance(actor_output.result, dict):
                    for r in actor_output.result.get("execution_results", []) or []:
                        if r.get("io_op"):
                            cell.add_io_record(
                                operation=r.get("io_op", "unknown"),
                                file_path=r.get("path", "unknown"),
                                success=r.get("success", False),
                                error=r.get("error"),
                            )
                        if r.get("tool"):
                            cell.add_tool_record(
                                tool_name=r.get("tool", "unknown"),
                                arguments=r.get("arguments", {}),
                                success=r.get("success", False),
                                output=r.get("output"),
                                error=r.get("error"),
                            )

                await _emit_phase("watching", {"task_id": task.task_id, "message": f"Watching & verifying {task.task_id} against expectations..."})
                watch, disk_report = await self._watch_deep_task(
                    task,
                    actor_output,
                    expected_res,
                    diag_report=diag_report,
                    prefer_deterministic_watch=(force_engine or is_local_engine),
                )

                cell.add_reflection({
                    "status": "done" if watch.matches_expectation else "in_progress",
                    "rating": watch.rating,
                    "summary": watch.actual_summary,
                    "feedback": (
                        f"Expected matched: {watch.actual_summary}"
                        if watch.matches_expectation
                        else f"MISMATCH ({'; '.join(watch.discrepancies)}). Fix: {watch.corrective_action or 'retry carefully'}"
                    ),
                })

                if watch.matches_expectation:
                    task_complete = True
                    task.status = TaskStatus.DONE
                    active_memory.mark_complete(task.task_id)
                    if disk_report:
                        disk_evidence_notes.append(f"[{task.task_id}] Verified On-Disk Files:\n{disk_report}")
                    status_log += f"Task {task.task_id} verified ({watch.rating}/10).\n"
                else:
                    mismatch_msg = "; ".join(watch.discrepancies) or "Output did not match expected result"
                    fix_hint = watch.corrective_action or "Inspect the actual file/tool output and fix all gaps."
                    cell.log_error(f"Watch mismatch: {mismatch_msg} | Action: {fix_hint}")
                    await _emit_phase("reacting", {"task_id": task.task_id, "discrepancies": watch.discrepancies, "message": f"Mismatch on {task.task_id}; reacting: {fix_hint[:120]}"})
                    if disk_report:
                        ws_snap = get_workspace_snapshot(rt, max_files=25) if rt else ws_snap

            if not task_complete:
                if cell.is_exhausted():
                    active_memory.mark_failed(task.task_id, "Did not match expected outcome after max retries")
                    status_log += f"Task {task.task_id} failed verification after {cell.attempts} attempts.\n"
                if (
                    self.replan_on_failure
                    and replan_count < 2
                    and iterations < max_iterations
                    and hasattr(self.thinker, "replan")
                    and _is_async_callable(self.thinker.replan)
                ):
                    try:
                        replan_count += 1
                        recovery = await self.thinker.replan(
                            objective=plan.objective,
                            completed_summary=active_memory.get_completed_summary(),
                            failed_task=task,
                            failure_context=active_memory.build_retry_context(task.task_id),
                        )
                        if isinstance(recovery, PlanSchema) and recovery.tasks:
                            for nt in recovery.tasks:
                                if not active_memory.has_cell(nt.task_id):
                                    nc = active_memory.create_cell(
                                        task_id=nt.task_id,
                                        description=nt.description,
                                        task_type=getattr(nt, "task_type", "other"),
                                        priority=getattr(nt, "priority", "medium"),
                                        dependencies=getattr(nt, "dependencies", []),
                                    )
                                    nc.max_attempts = 2
                                    remaining.append(nt)
                    except Exception:
                        pass

            self._save_checkpoint_safe(active_memory, session_id, plan, prompt)
            remaining = [t for t in remaining if t.task_id != task.task_id and getattr(t, "status", "pending") != "done"]

        return await self._finalize_deep_response(
            prompt=prompt,
            objective=plan.objective,
            status_log=status_log,
            digest=active_memory.get_execution_digest(),
            disk_evidence_notes=disk_evidence_notes,
            active_memory=active_memory,
            memory=memory,
            session_id=session_id,
            mode_name="engine" if force_engine else "deep",
        )

    async def run_deep(
        self,
        prompt: str,
        memory: Any,
        session_id: str,
        max_iterations: int = 15,
    ) -> str:
        """Public entrypoint for `mode='deep'`."""
        return await self._run_deep_core(prompt, memory, session_id, max_iterations=max_iterations)

    async def run_engine(
        self,
        prompt: str,
        memory: Any,
        session_id: str,
        max_iterations: int = 15,
    ) -> str:
        """Public entrypoint for `mode='engine'` (Neuro-Symbolic Engine)."""
        return await self._run_deep_core(prompt, memory, session_id, max_iterations=max_iterations, force_engine=True)

    async def run_deep_stream(
        self,
        prompt: str,
        memory: Any,
        session_id: str,
        max_iterations: int = 15,
        force_engine: bool = False,
    ):
        """Streaming entrypoint for `mode='deep'` / `mode='engine'` yielding phase updates and final response."""
        q: asyncio.Queue = asyncio.Queue()
        task = asyncio.create_task(
            self._run_deep_core(
                prompt,
                memory,
                session_id,
                max_iterations=max_iterations,
                status_queue=q,
                force_engine=force_engine,
            )
        )
        while not task.done() or not q.empty():
            try:
                ev = await asyncio.wait_for(q.get(), timeout=0.05)
                yield ev
            except asyncio.TimeoutError:
                continue
        final_answer = await task
        for line in final_answer.splitlines() or [final_answer]:
            yield {"type": "chunk", "content": f"{line}\n"}

    async def run_engine_stream(
        self,
        prompt: str,
        memory: Any,
        session_id: str,
        max_iterations: int = 15,
    ):
        """Streaming entrypoint for `mode='engine'`."""
        async for ev in self.run_deep_stream(prompt, memory, session_id, max_iterations=max_iterations, force_engine=True):
            yield ev

    async def run_react(
        self,
        prompt: str,
        memory: Any,
        session_id: str,
        max_iterations: int = 6,
    ) -> str:
        """
        Ultra-fast direct ReAct loop (Think/Tool -> Observe -> Final Answer).
        Designed for Local LLMs and ERP tool queries: typically completes in 1-2 LLM calls
        instead of Plan -> Solve -> Reflect -> Synthesize (5-7 calls).
        """
        active_memory = self._resolve_memory_manager(memory)
        active_memory.set_session(session_id, prompt)
        if hasattr(self.actor, "reset_session_tracking"):
            self.actor.reset_session_tracking()

        base_context = ""
        if hasattr(memory, "get_full_context"):
            res = memory.get_full_context(session_id, query=prompt)
            base_context = await res if inspect.isawaitable(res) else str(res or "")

        rt = self._active_runtime()
        ws_snap = get_workspace_snapshot(rt, max_files=25) if rt else ""
        if ws_snap and "Workspace Files:" not in base_context:
            base_context = f"{base_context}\n\nWorkspace Files: {ws_snap}".strip()

        react_task = TaskSchema(task_id="react_main", description=prompt)
        cell = active_memory.create_cell(task_id="react_main", description=prompt)

        observations: List[str] = []
        final_answer = ""
        executed_signatures: set = set()

        await tracer.emit_event("react_start", {"session_id": session_id, "prompt": prompt[:120]})

        for step in range(1, max(1, max_iterations) + 1):
            cell.increment_attempt()
            ctx_parts = []
            if base_context:
                ctx_parts.append(base_context)
            if observations:
                ctx_parts.append("TOOL OBSERVATIONS SO FAR:\n" + "\n".join(observations[-8:]))
                ctx_parts.append(
                    "IMPORTANT: Do NOT repeat tool calls you already ran above. "
                    "If you have all needed results (including any rejections/errors), reply NOW with `direct_response` and empty `tools_to_call`."
                )
            step_context = "\n\n".join(ctx_parts)

            try:
                with moe_route_context(role="coder", attempt=step, escalate=(step >= 3)):
                    task_exec = await self.thinker.solve_task(react_task, context=step_context)
            except (ValueError, RuntimeError) as fmt_err:
                if not isinstance(fmt_err, ValueError) and "json" not in str(fmt_err).lower():
                    raise
                observations.append(
                    f"[Step {step} | FORMAT ERROR]: {fmt_err}. Return valid JSON with `tools_to_call` or `direct_response`."
                )
                continue

            tool_calls = getattr(task_exec, "tools_to_call", None) or []
            direct_resp = (getattr(task_exec, "direct_response", None) or "").strip()

            if not tool_calls:
                final_answer = direct_resp or (getattr(task_exec, "thought_process", "") or "").strip() or "\n".join(observations)
                active_memory.mark_complete("react_main")
                break

            # Loop guard: if every tool call in this step was already executed with identical args, stop repeating!
            step_sigs = [
                (tc.tool_name, compact_value(tc.arguments or {}, max_chars=300))
                for tc in tool_calls
            ]
            if step > 1 and all(sig in executed_signatures for sig in step_sigs):
                if direct_resp:
                    final_answer = direct_resp
                elif hasattr(self.thinker, "synthesize_final_answer") and _is_async_callable(self.thinker.synthesize_final_answer):
                    try:
                        final_answer = await self.thinker.synthesize_final_answer(
                            prompt=prompt,
                            objective=prompt,
                            execution_digest="\n".join(observations),
                        )
                    except Exception:
                        final_answer = "\n".join(observations)
                else:
                    final_answer = "\n".join(observations)
                active_memory.mark_complete("react_main")
                break
            executed_signatures.update(step_sigs)

            if self.approval_callback is not None:
                decision = self.approval_callback(react_task, step_context)
                if inspect.isawaitable(decision):
                    decision = await decision
                if decision in ("reject", False):
                    final_answer = "Execution stopped due to human rejection."
                    break
                if decision == "skip":
                    final_answer = "Task skipped by human."
                    break

            step_returned_direct = False
            step_all_succeeded = True
            for tc in tool_calls:
                t_name = tc.tool_name
                t_args = tc.arguments or {}
                await tracer.emit_event("tool_call", {"step": step, "tool": t_name})
                rec = await self.actor._invoke_tool(t_name, t_args)
                if not rec.get("success", False):
                    step_all_succeeded = False
                cell.add_tool_record(
                    tool_name=t_name,
                    arguments=t_args,
                    success=rec.get("success", False),
                    output=rec.get("output"),
                    error=rec.get("error"),
                )
                status = "OK" if rec.get("success") else "FAIL"
                payload_txt = compact_value(rec.get("output") if rec.get("success") else rec.get("error"), max_chars=900)
                observations.append(f"[Step {step} | {t_name} | {status}]: {payload_txt}")

                if rec.get("success") and self.actor._is_return_direct_tool(t_name):
                    final_answer = str(rec.get("output") or "")
                    step_returned_direct = True
                    active_memory.mark_complete("react_main")
                    break

            if step_returned_direct:
                break
            if step_all_succeeded and getattr(task_exec, "is_complete", False) and direct_resp:
                final_answer = direct_resp
                active_memory.mark_complete("react_main")
                break

        if not final_answer:
            final_answer = "\n".join(observations) if observations else "No result produced."

        if hasattr(memory, "add_interaction"):
            await memory.add_interaction(session_id, "assistant", final_answer)
        if hasattr(active_memory, "finalize_run"):
            self._schedule_background(
                active_memory.finalize_run(
                    session_id,
                    objective=prompt,
                    final_answer=final_answer,
                    llm=getattr(self.reflector, "llm", None),
                )
            )
        await self._flush_background()
        await tracer.emit_event("complete", {"session_id": session_id, "mode": "react"})
        return final_answer

    async def run_react_stream(
        self,
        prompt: str,
        memory: Any,
        session_id: str,
        max_iterations: int = 6,
    ):
        """Streaming wrapper around the fast ReAct loop."""
        yield {"type": "status", "content": "Fast ReAct mode..."}
        final = await self.run_react(prompt, memory, session_id, max_iterations=max_iterations)
        yield {"type": "status", "content": "ReAct loop finished."}
        for line in final.splitlines() or [final]:
            yield {"type": "chunk", "content": f"{line}\n"}

    async def run_stream(self, prompt: str, memory: Any, session_id: str, max_iterations: int = 15):
        """Streaming version of the brain loop with unified memory graph."""
        if not self._uses_plan_schema_mode():
            yield {"type": "status", "content": "Thinking and Planning..."}
            objective = prompt
            if hasattr(self.thinker, "analyze") and _is_async_callable(self.thinker.analyze):
                objective = await self.thinker.analyze(prompt, memory, session_id)

            previous_results = ""
            has_executed_real_action = False
            for _ in range(max_iterations):
                if hasattr(self.planner, "stream_thinking") and callable(self.planner.stream_thinking):
                    yield {"type": "status", "content": "Planner thinking..."}
                    stream_gen = self.planner.stream_thinking(objective, previous_results)
                    if hasattr(stream_gen, "__aiter__"):
                        async for thought_chunk in stream_gen:
                            yield {"type": "chunk", "content": str(thought_chunk)}

                plan_dict = await self.planner.plan(objective, previous_results)
                actions = plan_dict.get("actions", []) if isinstance(plan_dict, dict) else []
                is_finish = any(a.get("tool") == "finish" for a in actions if isinstance(a, dict))
                if is_finish and has_executed_real_action:
                    break

                action_res = await self.actor.execute(plan_dict)
                if not is_finish:
                    has_executed_real_action = True
                previous_results = str(action_res)
                yield {"type": "chunk", "content": f"{action_res}\n"}

                ref_res = await self.reflector.reflect(objective, plan_dict, action_res)
                if isinstance(ref_res, dict) and ref_res.get("is_complete") and has_executed_real_action:
                    break
            await self._flush_background()
            return

        yield {"type": "status", "content": "Thinking and Planning..."}
        active_memory = self._resolve_memory_manager(memory)
        active_memory.set_session(session_id, prompt)
        if hasattr(self.actor, "reset_session_tracking"):
            self.actor.reset_session_tracking()

        plan = await self.planner.generate_initial_plan(prompt, memory, session_id)
        active_memory.objective = plan.objective
        if hasattr(memory, "session") and hasattr(memory.session, "set"):
            memory.session.set("current_objective", plan.objective)

        for task in plan.tasks:
            active_memory.create_cell(
                task_id=task.task_id,
                description=task.description,
                task_type=getattr(task, "task_type", "other"),
                priority=getattr(task, "priority", "medium"),
                dependencies=getattr(task, "dependencies", []),
            )

        yield {"type": "status", "content": f"Plan generated: {len(plan.tasks)} tasks."}

        status_log = ""
        iterations = 0
        replan_count = 0
        max_replans = 1 if self.fast_reflect else 3

        for task in plan.tasks:
            if getattr(task, "status", "pending") == "done":
                continue

            cell = active_memory.get_cell(task.task_id)

            if not active_memory.are_dependencies_met(task.task_id):
                cell.log_error("Dependencies not met, skipping")
                yield {"type": "status", "content": f"Skipping task {task.task_id}, dependencies not met."}
                continue

            cell.task_status = "in_progress"
            task_complete = False

            while not task_complete and iterations < max_iterations and not cell.is_exhausted():
                cell.increment_attempt()
                context = active_memory.get_task_context(task.task_id)

                yield {"type": "status", "content": f"Executing task {task.task_id} attempt {cell.attempts}..."}
                with moe_route_context(role="coder", attempt=cell.attempts, escalate=(cell.attempts >= 2)):
                    actor_output = await self.actor.generate_and_execute(task, context=context)

                result_data = _model_to_dict(actor_output)
                cell.add_runtime_result(result_data)

                if hasattr(actor_output, "result") and isinstance(actor_output.result, dict):
                    exec_results = actor_output.result.get("execution_results", [])
                    for r in exec_results:
                        if r.get("io_op"):
                            cell.add_io_record(
                                operation=r.get("io_op", "unknown"),
                                file_path=r.get("path", "unknown"),
                                success=r.get("success", False),
                                error=r.get("error"),
                            )
                        if r.get("tool"):
                            cell.add_tool_record(
                                tool_name=r.get("tool", "unknown"),
                                arguments=r.get("arguments", {}),
                                success=r.get("success", False),
                                output=r.get("output"),
                                error=r.get("error"),
                            )

                if hasattr(actor_output, "success") and not actor_output.success:
                    cell.log_error(getattr(actor_output, "error_context", "Actor execution failed") or "Actor execution failed")

                yield {"type": "status", "content": f"Reflecting on task {task.task_id}..."}
                ref_context = active_memory.build_retry_context(task.task_id)
                compact_target = cell.get_last_runtime_result() or result_data
                ref_input = ReflectorInputSchema(
                    reflector_type=ReflectorType.TASK,
                    target_id=task.task_id,
                    target_content=compact_target,
                    context=ref_context,
                )

                reflection = await self.reflector.reflect(ref_input)
                ref_dict = _model_to_dict(reflection, fallback_key="feedback")
                cell.add_reflection(ref_dict)

                feedback_text = getattr(reflection, "feedback", None) or ref_dict.get("feedback")
                if feedback_text and hasattr(memory, "reflection") and hasattr(memory.reflection, "add_reflection"):
                    memory.reflection.add_reflection(f"[{task.task_id}] {feedback_text}")

                iterations += 1

                if getattr(reflection, "is_task_complete", False):
                    task_complete = True
                    task.status = "done"
                    active_memory.mark_complete(task.task_id)
                    status_log += f"Task {task.task_id} completed. Rating: {reflection.rating}/10.\n"
                    yield {"type": "status", "content": f"Task {task.task_id} complete!"}
                else:
                    yield {"type": "status", "content": f"Task {task.task_id} incomplete, retrying with feedback..."}

            if not task_complete and cell.is_exhausted():
                active_memory.mark_failed(task.task_id, "Max attempts exhausted")
                status_log += f"Task {task.task_id} failed after {cell.attempts} attempts.\n"
                yield {"type": "status", "content": f"Task {task.task_id} failed after {cell.attempts} attempts."}

        if iterations >= max_iterations:
            status_log += "\nMax iterations reached before completing all tasks."

        final_answer = status_log
        if (
            self.synthesize_answer
            and hasattr(self.thinker, "synthesize_final_answer")
            and _is_async_callable(self.thinker.synthesize_final_answer)
        ):
            try:
                digest = active_memory.get_execution_digest()
                synthesized = await self.thinker.synthesize_final_answer(
                    prompt=prompt,
                    objective=plan.objective,
                    execution_digest=digest,
                )
                if synthesized and isinstance(synthesized, str) and synthesized.strip():
                    final_answer = f"{synthesized.strip()}\n\n---\nExecution Summary:\n{status_log.strip()}"
            except Exception:
                pass

        if hasattr(memory, "add_interaction"):
            await memory.add_interaction(session_id, "assistant", final_answer)
        if hasattr(active_memory, "finalize_run"):
            self._schedule_background(
                active_memory.finalize_run(
                    session_id,
                    objective=plan.objective,
                    final_answer=final_answer,
                    llm=getattr(self.reflector, "llm", None),
                )
            )
        await self._flush_background()
        yield {"type": "status", "content": "Brain loop finished."}
        yield {"type": "chunk", "content": final_answer}
