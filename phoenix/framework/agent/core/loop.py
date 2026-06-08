import asyncio
from typing import Dict, Any, Optional

from phoenix.framework.agent.cognition.thinker import Thinker
from phoenix.framework.agent.cognition.planner import Planner
from phoenix.framework.agent.cognition.reflector import Reflector
from phoenix.framework.agent.cognition.analyzer import Analyzer
from phoenix.framework.agent.cognition.actor import Actor
from phoenix.framework.agent.cognition.reflector.schema import ReflectorInputSchema, ReflectorType
from phoenix.framework.agent.memory.manager import MemoryManager


class AgentLoop:
    """Coordinates the autonomous brain loop using a unified memory graph"""

    def __init__(self, thinker: Thinker, planner: Planner, actor: Actor, reflector: Reflector, analyzer: Analyzer):
        self.thinker = thinker
        self.planner = planner
        self.actor = actor
        self.reflector = reflector
        self.analyzer = analyzer
        self.memory_manager = MemoryManager()
        self._background_tasks = set()

    def _schedule_background(self, coro):
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def run(self, prompt: str, memory, session_id: str, max_iterations: int = 15) -> str:
        """Execute the full brain loop with unified memory graph"""

        plan = await self.planner.generate_initial_plan(prompt, memory, session_id)
        self.memory_manager.set_session(session_id, plan.objective)
        memory.session.set("current_objective", plan.objective)

        for task in plan.tasks:
            cell = self.memory_manager.create_cell(
                task_id=task.task_id,
                description=task.description,
                task_type=getattr(task, "task_type", "other"),
                priority=getattr(task, "priority", "medium"),
                dependencies=getattr(task, "dependencies", [])
            )

        final_answer = ""
        iterations = 0

        for task in plan.tasks:
            if task.status == "done":
                continue

            cell = self.memory_manager.get_cell(task.task_id)

            if not self.memory_manager.are_dependencies_met(task.task_id):
                cell.log_error("Dependencies not met, skipping")
                continue

            cell.task_status = "in_progress"
            task_complete = False

            problems = await self.planner.define_task_problems(task)
            if hasattr(problems, "problems"):
                for p in problems.problems:
                    cell.add_problem(p.dict() if hasattr(p, "dict") else dict(p))

            while not task_complete and iterations < max_iterations and not cell.is_exhausted():
                cell.increment_attempt()

                context = self.memory_manager.get_full_context(task.task_id)

                actor_output = await self.actor.generate_and_execute(problems, context=context)

                result_data = actor_output.dict() if hasattr(actor_output, "dict") else {"output": str(actor_output)}
                cell.add_runtime_result(result_data)

                if hasattr(actor_output, "result") and isinstance(actor_output.result, dict):
                    exec_results = actor_output.result.get("execution_results", [])
                    for r in exec_results:
                        if r.get("io_op"):
                            cell.add_io_record(
                                operation=r.get("io_op", "unknown"),
                                file_path=r.get("path", "unknown"),
                                success=r.get("success", False),
                                error=r.get("error")
                            )
                        if r.get("tool"):
                            cell.add_tool_record(
                                tool_name=r.get("tool", "unknown"),
                                arguments={},
                                success=r.get("success", False),
                                output=r.get("output"),
                                error=r.get("error")
                            )

                if hasattr(actor_output, "success") and not actor_output.success:
                    cell.log_error(getattr(actor_output, "error_context", "Actor execution failed"))

                ref_context = self.memory_manager.build_retry_context(task.task_id)
                ref_input = ReflectorInputSchema(
                    reflector_type=ReflectorType.TASK,
                    target_id=task.task_id,
                    target_content=result_data,
                    context=ref_context
                )

                reflection = await self.reflector.reflect(ref_input)
                cell.add_reflection(reflection.dict() if hasattr(reflection, "dict") else {"feedback": str(reflection)})

                self._schedule_background(
                    memory.add_interaction(session_id, "system", f"Task {task.task_id} attempt {cell.attempts}. Success: {getattr(actor_output, 'success', False)}")
                )

                iterations += 1

                if reflection.is_task_complete:
                    task_complete = True
                    task.status = "done"
                    self.memory_manager.mark_complete(task.task_id)
                    final_answer += f"Task {task.task_id} completed. Rating: {reflection.rating}/10.\n"

            if not task_complete:
                if cell.is_exhausted():
                    self.memory_manager.mark_failed(task.task_id, "Max attempts exhausted")
                    final_answer += f"Task {task.task_id} failed after {cell.attempts} attempts.\n"

        if iterations >= max_iterations:
            final_answer += "\nMax iterations reached before completing all tasks."

        await memory.add_interaction(session_id, "assistant", final_answer)
        return final_answer

    async def run_stream(self, prompt: str, memory, session_id: str, max_iterations: int = 15):
        """Streaming version of the brain loop with unified memory graph"""

        yield {"type": "status", "content": "Thinking and Planning..."}
        plan = await self.planner.generate_initial_plan(prompt, memory, session_id)
        self.memory_manager.set_session(session_id, plan.objective)
        memory.session.set("current_objective", plan.objective)

        for task in plan.tasks:
            self.memory_manager.create_cell(
                task_id=task.task_id,
                description=task.description,
                task_type=getattr(task, "task_type", "other"),
                priority=getattr(task, "priority", "medium"),
                dependencies=getattr(task, "dependencies", [])
            )

        yield {"type": "status", "content": f"Plan generated: {len(plan.tasks)} tasks."}

        final_answer = ""
        iterations = 0

        for task in plan.tasks:
            if task.status == "done":
                continue

            cell = self.memory_manager.get_cell(task.task_id)

            if not self.memory_manager.are_dependencies_met(task.task_id):
                cell.log_error("Dependencies not met, skipping")
                yield {"type": "status", "content": f"Skipping task {task.task_id}, dependencies not met."}
                continue

            cell.task_status = "in_progress"
            task_complete = False

            yield {"type": "status", "content": f"Defining problems for task {task.task_id}..."}
            problems = await self.planner.define_task_problems(task)
            if hasattr(problems, "problems"):
                for p in problems.problems:
                    cell.add_problem(p.dict() if hasattr(p, "dict") else dict(p))

            while not task_complete and iterations < max_iterations and not cell.is_exhausted():
                cell.increment_attempt()
                context = self.memory_manager.get_full_context(task.task_id)

                yield {"type": "status", "content": f"Executing task {task.task_id} attempt {cell.attempts}..."}
                actor_output = await self.actor.generate_and_execute(problems, context=context)

                result_data = actor_output.dict() if hasattr(actor_output, "dict") else {"output": str(actor_output)}
                cell.add_runtime_result(result_data)

                if hasattr(actor_output, "result") and isinstance(actor_output.result, dict):
                    exec_results = actor_output.result.get("execution_results", [])
                    for r in exec_results:
                        if r.get("io_op"):
                            cell.add_io_record(
                                operation=r.get("io_op", "unknown"),
                                file_path=r.get("path", "unknown"),
                                success=r.get("success", False),
                                error=r.get("error")
                            )
                        if r.get("tool"):
                            cell.add_tool_record(
                                tool_name=r.get("tool", "unknown"),
                                arguments={},
                                success=r.get("success", False),
                                output=r.get("output"),
                                error=r.get("error")
                            )

                if hasattr(actor_output, "success") and not actor_output.success:
                    cell.log_error(getattr(actor_output, "error_context", "Actor execution failed"))

                yield {"type": "status", "content": f"Reflecting on task {task.task_id}..."}
                ref_context = self.memory_manager.build_retry_context(task.task_id)
                ref_input = ReflectorInputSchema(
                    reflector_type=ReflectorType.TASK,
                    target_id=task.task_id,
                    target_content=result_data,
                    context=ref_context
                )

                reflection = await self.reflector.reflect(ref_input)
                cell.add_reflection(reflection.dict() if hasattr(reflection, "dict") else {"feedback": str(reflection)})

                self._schedule_background(
                    memory.add_interaction(session_id, "system", f"Task {task.task_id} attempt {cell.attempts}. Success: {getattr(actor_output, 'success', False)}")
                )

                iterations += 1

                if reflection.is_task_complete:
                    task_complete = True
                    task.status = "done"
                    self.memory_manager.mark_complete(task.task_id)
                    final_answer += f"Task {task.task_id} completed. Rating: {reflection.rating}/10.\n"
                    yield {"type": "status", "content": f"Task {task.task_id} complete!"}
                else:
                    yield {"type": "status", "content": f"Task {task.task_id} incomplete, retrying with feedback..."}

            if not task_complete:
                if cell.is_exhausted():
                    self.memory_manager.mark_failed(task.task_id, "Max attempts exhausted")
                    final_answer += f"Task {task.task_id} failed after {cell.attempts} attempts.\n"
                    yield {"type": "status", "content": f"Task {task.task_id} failed after {cell.attempts} attempts."}

        if iterations >= max_iterations:
            final_answer += "\nMax iterations reached before completing all tasks."

        await memory.add_interaction(session_id, "assistant", final_answer)
        yield {"type": "status", "content": "Brain loop finished."}
        yield {"type": "chunk", "content": final_answer}
