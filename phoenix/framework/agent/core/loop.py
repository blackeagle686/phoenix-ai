import asyncio
import json
from pathlib import Path
from typing import Dict, Any, Optional

from phoenix.framework.agent.cognition.thinker import Thinker
from phoenix.framework.agent.cognition.planner import Planner
from phoenix.framework.agent.cognition.reflector import Reflector
from phoenix.framework.agent.cognition.analyzer import Analyzer
from phoenix.framework.agent.cognition.actor import Actor

from phoenix.framework.agent.cognition.utils import generate_unique_id
from phoenix.framework.agent.cognition.pipeline import BrainRegistry, CognitionPipeline

from phoenix.framework.agent.cognition.actor.schema import ActorInputSchema
from phoenix.framework.agent.cognition.planner.schema import TaskType
from phoenix.framework.agent.cognition.reflector.schema import ReflectorInputSchema, ReflectorType

class AgentLoop:
    """
    Coordinates the autonomous workflow with a parallel Async Channel Architecture:
    Bootstrap(Think/Analyze) -> [ Planner Queue <-> Actor Queue <-> Reflector Queue ]
    """
    def __init__(self, thinker: Thinker, planner: Planner, actor: Actor, reflector: Reflector, analyzer: Analyzer):
        self.thinker = thinker
        self.planner = planner
        self.actor = actor
        self.reflector = reflector
        self.analyzer = analyzer
        self._background_tasks = set()
        
        # We retain BrainRegistry/CognitionPipeline ONLY for the synchronous Bootstrap phase
        self._registry = BrainRegistry()
        self._bootstrap_pipeline = self._load_pipeline_spec("default_bootstrap_pipeline.json")
        self._register_bootstrap_brains()

    def _load_pipeline_spec(self, filename: str) -> CognitionPipeline:
        spec_path = Path(__file__).resolve().parents[1] / "cognition" / "pipelines" / filename
        return CognitionPipeline.from_json_file(str(spec_path))

    def _register_bootstrap_brains(self):
        async def thinker_handler(data):
            objective = await self.thinker.analyze(data["prompt"], data["memory"], data["session_id"])
            return {"objective": objective}

        async def analyzer_handler(data):
            root_dir = data.get("root_dir", ".")
            analysis = await self.analyzer.analyze_workspace(data["prompt"], root_dir=root_dir)
            return {"analysis": analysis}

        self._registry.register("thinker", thinker_handler)
        self._registry.register("analyzer", analyzer_handler)

    def set_pipeline_specs(self, bootstrap_pipeline_path: Optional[str] = None):
        """
        Replace default JSON bootstrap pipeline with user-provided spec.
        """
        if bootstrap_pipeline_path:
            self._bootstrap_pipeline = CognitionPipeline.from_json_file(bootstrap_pipeline_path)

    def _schedule_background(self, coro):
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def _bootstrap(self, prompt: str, memory, session_id: str) -> tuple[str, str]:
        bootstrap_state = await self._bootstrap_pipeline.run(
            self._registry,
            {
                "prompt": prompt,
                "memory": memory,
                "session_id": session_id,
                "root_dir": "."
            }
        )
        thinker_out = bootstrap_state.get("thinker_output", {})
        objective = thinker_out.get("objective", prompt) if isinstance(thinker_out, dict) else prompt
        
        analyzer_out = bootstrap_state.get("analyzer_output", {})
        analysis = analyzer_out.get("analysis", {}) if isinstance(analyzer_out, dict) else {}

        task_file_id = generate_unique_id()
        memory.session.set("current_objective", objective)
        memory.session.set("project_analysis", analysis)
        memory.session.set("task_file_id", task_file_id)
        
        return objective, task_file_id

    async def run(self, prompt: str, memory, session_id: str, max_iterations: int = 5) -> str:
        objective, task_file_id = await self._bootstrap(prompt, memory, session_id)
        
        planner_queue = asyncio.Queue()
        actor_queue = asyncio.Queue()
        reflector_queue = asyncio.Queue()
        completion_event = asyncio.Event()
        
        shared_state = {
            "final_answer": "",
            "previous_results": "",
            "iterations": 0
        }

        # --- Worker 1: Planner ---
        async def _planner_worker():
            try:
                while not completion_event.is_set():
                    await planner_queue.get()
                    
                    if shared_state["iterations"] >= max_iterations:
                        shared_state["final_answer"] = "Maximum iterations reached. Forcing completion."
                        completion_event.set()
                        planner_queue.task_done()
                        break

                    plan_output = await self.planner.plan(
                        objective,
                        task_file_id=task_file_id,
                        previous_results=shared_state["previous_results"]
                    )
                    
                    shared_state["iterations"] += 1
                    actions = plan_output.get("actions", [])
                    if not actions and "tool" in plan_output:
                        actions = [{"tool": plan_output["tool"], "kwargs": plan_output.get("kwargs", {})}]
                        
                    has_finish = any(a.get("tool") == "finish" for a in actions)
                    
                    if has_finish:
                        shared_state["final_answer"] = shared_state["previous_results"] or "Task completed by Planner."
                        completion_event.set()
                        planner_queue.task_done()
                        break
                        
                    if not actions:
                        # Planner stalled, trigger it again or abort
                        await asyncio.sleep(1)
                        await planner_queue.put("retry")
                    else:
                        for action in actions:
                            tool_name = action.get("tool")
                            if tool_name == "finish":
                                continue
                            task_input = ActorInputSchema(
                                task_id=generate_unique_id(),
                                task_type=TaskType.OTHER,
                                tool_name=tool_name,
                                payload=action.get("kwargs", {})
                            )
                            await actor_queue.put(task_input)
                            
                    planner_queue.task_done()
            except asyncio.CancelledError:
                pass
            except Exception as e:
                shared_state["final_answer"] = f"Planner Fault: {e}"
                completion_event.set()

        # --- Worker 2: Actor Pool ---
        async def _actor_worker():
            try:
                while not completion_event.is_set():
                    task_input = await actor_queue.get()
                    # Execute purely on input. Reflection is deferred to the Reflector Queue
                    actor_output = await self.actor.execute(task_input, task_context=None)
                    await reflector_queue.put((task_input, actor_output))
                    actor_queue.task_done()
            except asyncio.CancelledError:
                pass

        # --- Worker 3: Reflector ---
        async def _reflector_worker():
            try:
                while not completion_event.is_set():
                    task_input, actor_output = await reflector_queue.get()
                    
                    ref_input = ReflectorInputSchema(
                        reflector_type=ReflectorType.TASK,
                        target_id=task_input.task_id,
                        target_content={"action": task_input.dict(), "result": actor_output.dict()},
                        context=objective
                    )
                    
                    reflection = await self.reflector.reflect(ref_input)
                    
                    result_summary = f"\nAction: `{task_input.tool_name}`\nResult Status: {actor_output.success}\nFeedback: {reflection.feedback}\nRating: {reflection.rating}/10\n"
                    shared_state["previous_results"] += result_summary
                    
                    self._schedule_background(memory.add_interaction(session_id, "system", f"Tool Output: {actor_output.success}"))
                    self._schedule_background(memory.long_term.add(session_id, reflection.feedback))
                    
                    # Trigger the next planning cycle now that results are appended
                    await planner_queue.put("trigger_planning")
                    reflector_queue.task_done()
            except asyncio.CancelledError:
                pass

        # Spin up parallel tasks
        planner_task = asyncio.create_task(_planner_worker())
        # We can spin up multiple actor workers for high concurrency!
        actor_tasks = [asyncio.create_task(_actor_worker()) for _ in range(3)]
        reflector_task = asyncio.create_task(_reflector_worker())
        
        # Kick off the loop
        await planner_queue.put("start")
        
        # Wait for finish signal
        await completion_event.wait()
        
        # Cleanup
        planner_task.cancel()
        for t in actor_tasks:
            t.cancel()
        reflector_task.cancel()
        
        await memory.add_interaction(session_id, "assistant", shared_state["final_answer"])
        return shared_state["final_answer"]

    async def run_stream(self, prompt: str, memory, session_id: str, max_iterations: int = 5):
        """
        Streaming version leveraging the same Parallel Queue architecture.
        Yields UI updates into a stream_queue.
        """
        stream_queue = asyncio.Queue()
        
        await stream_queue.put({"type": "status", "content": "🤔 Initializing context..."})
        objective, task_file_id = await self._bootstrap(prompt, memory, session_id)
        
        planner_queue = asyncio.Queue()
        actor_queue = asyncio.Queue()
        reflector_queue = asyncio.Queue()
        completion_event = asyncio.Event()
        
        shared_state = {
            "final_answer": "",
            "previous_results": "",
            "iterations": 0
        }

        async def _planner_worker():
            try:
                while not completion_event.is_set():
                    await planner_queue.get()
                    
                    if shared_state["iterations"] >= max_iterations:
                        shared_state["final_answer"] = "Maximum iterations reached."
                        completion_event.set()
                        planner_queue.task_done()
                        break

                    await stream_queue.put({"type": "status", "content": f"🤔 Planner thinking (Iteration {shared_state['iterations'] + 1})..."})
                    
                    # Stream Planner thoughts directly to UI
                    async for thought in self.planner.stream_thinking(objective, task_file_id=task_file_id, previous_results=shared_state["previous_results"]):
                        await stream_queue.put({"type": "chunk", "content": thought})

                    plan_output = await self.planner.plan(
                        objective,
                        task_file_id=task_file_id,
                        previous_results=shared_state["previous_results"]
                    )
                    
                    shared_state["iterations"] += 1
                    actions = plan_output.get("actions", [])
                    if not actions and "tool" in plan_output:
                        actions = [{"tool": plan_output["tool"], "kwargs": plan_output.get("kwargs", {})}]
                        
                    has_finish = any(a.get("tool") == "finish" for a in actions)
                    
                    if has_finish:
                        shared_state["final_answer"] = shared_state["previous_results"] or "Done."
                        completion_event.set()
                        planner_queue.task_done()
                        break
                        
                    if actions:
                        await stream_queue.put({"type": "status", "content": "🛠️ Dispatching actions to Actor Pool..."})
                        for action in actions:
                            tool_name = action.get("tool")
                            if tool_name != "finish":
                                await actor_queue.put(ActorInputSchema(
                                    task_id=generate_unique_id(),
                                    task_type=TaskType.OTHER,
                                    tool_name=tool_name,
                                    payload=action.get("kwargs", {})
                                ))
                    else:
                        await asyncio.sleep(1)
                        await planner_queue.put("retry")
                    
                    planner_queue.task_done()
            except asyncio.CancelledError:
                pass
            except Exception as e:
                shared_state["final_answer"] = f"Planner Fault: {e}"
                completion_event.set()

        async def _actor_worker():
            try:
                while not completion_event.is_set():
                    task_input = await actor_queue.get()
                    actor_output = await self.actor.execute(task_input, task_context=None)
                    await stream_queue.put({"type": "status", "content": f"⚙️ Executed: {task_input.tool_name}"})
                    await reflector_queue.put((task_input, actor_output))
                    actor_queue.task_done()
            except asyncio.CancelledError:
                pass

        async def _reflector_worker():
            try:
                while not completion_event.is_set():
                    task_input, actor_output = await reflector_queue.get()
                    await stream_queue.put({"type": "status", "content": f"🧐 Reflecting on: {task_input.tool_name}..."})
                    
                    ref_input = ReflectorInputSchema(
                        reflector_type=ReflectorType.TASK,
                        target_id=task_input.task_id,
                        target_content={"action": task_input.dict(), "result": actor_output.dict()},
                        context=objective
                    )
                    
                    reflection = await self.reflector.reflect(ref_input)
                    shared_state["previous_results"] += f"\nAction: `{task_input.tool_name}`\nStatus: {actor_output.success}\nFeedback: {reflection.feedback}\n"
                    
                    self._schedule_background(memory.add_interaction(session_id, "system", f"Output: {actor_output.success}"))
                    self._schedule_background(memory.long_term.add(session_id, reflection.feedback))
                    
                    await planner_queue.put("trigger")
                    reflector_queue.task_done()
            except asyncio.CancelledError:
                pass

        workers = [
            asyncio.create_task(_planner_worker()),
            asyncio.create_task(_actor_worker()),
            asyncio.create_task(_actor_worker()), # Parallel execution concurrency
            asyncio.create_task(_reflector_worker())
        ]
        
        await planner_queue.put("start")
        
        while not completion_event.is_set():
            try:
                msg = await asyncio.wait_for(stream_queue.get(), timeout=0.2)
                yield msg
            except asyncio.TimeoutError:
                continue

        for w in workers:
            w.cancel()
            
        yield {"type": "status", "content": "✅ Finalizing..."}
        await memory.add_interaction(session_id, "assistant", shared_state["final_answer"])
        for chunk in shared_state["final_answer"].split():
            yield {"type": "chunk", "content": chunk + " "}
