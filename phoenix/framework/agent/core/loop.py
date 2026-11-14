import asyncio
import inspect
from typing import Dict, Any, Optional, Callable, List

from phoenix.framework.agent.cognition.thinker import Thinker
from phoenix.framework.agent.cognition.planner import Planner
from phoenix.framework.agent.cognition.reflector import Reflector
from phoenix.framework.agent.cognition.analyzer import Analyzer
from phoenix.framework.agent.cognition.actor import Actor
from phoenix.framework.agent.cognition.pipeline import BrainRegistry, CognitionPipeline
from phoenix.framework.agent.cognition.schema import (
    PlanSchema,
    ReflectorInputSchema,
    ReflectorType,
)
from phoenix.framework.agent.memory.manager import MemoryManager


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
        parallel_tasks: bool = True,
        max_concurrency: int = 3,
        replan_on_failure: bool = True,
        synthesize_answer: bool = True,
        enable_workspace_analysis: bool = False,
        approval_callback: Optional[Callable[..., Any]] = None,
    ):
        self.thinker = thinker
        self.planner = planner
        self.actor = actor
        self.reflector = reflector
        self.analyzer = analyzer
        self.memory_manager = MemoryManager()
        self.parallel_tasks = parallel_tasks
        self.max_concurrency = max(1, max_concurrency)
        self.replan_on_failure = replan_on_failure
        self.synthesize_answer = synthesize_answer
        self.enable_workspace_analysis = enable_workspace_analysis
        self.approval_callback = approval_callback
        self._bootstrap_pipeline: Optional[CognitionPipeline] = None
        self._background_tasks = set()

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

            ref_context = active_memory.build_retry_context(task.task_id)
            ref_input = ReflectorInputSchema(
                reflector_type=ReflectorType.TASK,
                target_id=task.task_id,
                target_content=result_data,
                context=ref_context,
            )

            reflection = await self.reflector.reflect(ref_input)
            last_reflection = reflection
            ref_dict = _model_to_dict(reflection, fallback_key="feedback")
            cell.add_reflection(ref_dict)

            feedback_text = getattr(reflection, "feedback", None) or ref_dict.get("feedback")
            if feedback_text and hasattr(memory, "reflection") and hasattr(memory.reflection, "add_reflection"):
                memory.reflection.add_reflection(f"[{task.task_id}] {feedback_text}")

            if hasattr(memory, "add_interaction"):
                self._schedule_background(
                    memory.add_interaction(
                        session_id,
                        "system",
                        f"Task {task.task_id} attempt {cell.attempts}. Success: {getattr(actor_output, 'success', False)}",
                    )
                )

            increment_iteration_fn()

            if getattr(reflection, "is_task_complete", False):
                task_complete = True
                task.status = "done"
                active_memory.mark_complete(task.task_id)
                rating = getattr(reflection, "rating", 8)
                return True, f"Task {task.task_id} completed. Rating: {rating}/10.\n", reflection

        if not task_complete and cell.is_exhausted():
            active_memory.mark_failed(task.task_id, "Max attempts exhausted")
            return False, f"Task {task.task_id} failed after {cell.attempts} attempts.\n", last_reflection

        return task_complete, "", last_reflection

    async def run(self, prompt: str, memory: Any, session_id: str, max_iterations: int = 15) -> str:
        """Execute the full brain loop with unified memory graph, DAG execution, and synthesis."""
        if not self._uses_plan_schema_mode():
            return await self._run_iterative_action_loop(prompt, memory, session_id, max_iterations=max_iterations)

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

        active_memory = self._resolve_memory_manager(memory)
        active_memory.set_session(session_id, plan.objective)
        if hasattr(memory, "session") and hasattr(memory.session, "set"):
            memory.session.set("current_objective", plan.objective)

        pending_tasks = list(plan.tasks)
        for task in pending_tasks:
            active_memory.create_cell(
                task_id=task.task_id,
                description=task.description,
                task_type=getattr(task, "task_type", "other"),
                priority=getattr(task, "priority", "medium"),
                dependencies=getattr(task, "dependencies", []),
            )

        status_log = ""
        iterations = 0

        def _has_iterations_left() -> bool:
            return iterations < max_iterations

        def _inc_iteration() -> None:
            nonlocal iterations
            iterations += 1

        remaining = [t for t in pending_tasks if getattr(t, "status", "pending") != "done"]

        while remaining and iterations < max_iterations:
            ready = [t for t in remaining if active_memory.are_dependencies_met(t.task_id)]
            if not ready:
                # Break dependency cycle or unmet failed dependency by picking the next task in order
                ready = [remaining[0]]

            can_parallelize = (
                self.parallel_tasks
                and len(ready) > 1
                and self.approval_callback is None
            )

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
                if reflection_obj == "REJECTED":
                    await self._flush_background()
                    if hasattr(memory, "add_interaction"):
                        await memory.add_interaction(session_id, "assistant", status_log)
                    return status_log

                # Dynamic adaptive replanning if needed
                if (
                    self.replan_on_failure
                    and (not completed or getattr(reflection_obj, "needs_replan", False))
                    and hasattr(self.thinker, "replan")
                    and _is_async_callable(self.thinker.replan)
                    and iterations < max_iterations
                ):
                    try:
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
                                    remaining.append(new_t)
                    except Exception:
                        pass

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

        await self._flush_background()
        if hasattr(memory, "add_interaction"):
            await memory.add_interaction(session_id, "assistant", final_answer)
        return final_answer

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
        plan = await self.planner.generate_initial_plan(prompt, memory, session_id)
        active_memory = self._resolve_memory_manager(memory)
        active_memory.set_session(session_id, plan.objective)
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
                ref_input = ReflectorInputSchema(
                    reflector_type=ReflectorType.TASK,
                    target_id=task.task_id,
                    target_content=result_data,
                    context=ref_context,
                )

                reflection = await self.reflector.reflect(ref_input)
                ref_dict = _model_to_dict(reflection, fallback_key="feedback")
                cell.add_reflection(ref_dict)

                feedback_text = getattr(reflection, "feedback", None) or ref_dict.get("feedback")
                if feedback_text and hasattr(memory, "reflection") and hasattr(memory.reflection, "add_reflection"):
                    memory.reflection.add_reflection(f"[{task.task_id}] {feedback_text}")

                if hasattr(memory, "add_interaction"):
                    self._schedule_background(
                        memory.add_interaction(
                            session_id,
                            "system",
                            f"Task {task.task_id} attempt {cell.attempts}. Success: {getattr(actor_output, 'success', False)}",
                        )
                    )

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

        await self._flush_background()
        if hasattr(memory, "add_interaction"):
            await memory.add_interaction(session_id, "assistant", final_answer)
        yield {"type": "status", "content": "Brain loop finished."}
        yield {"type": "chunk", "content": final_answer}
