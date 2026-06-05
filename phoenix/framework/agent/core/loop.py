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

    async def run(self, prompt: str, memory, session_id: str, max_iterations: int = 15) -> str:
        objective, task_file_id = await self._bootstrap(prompt, memory, session_id)
        
        planner_queue = asyncio.Queue()
        actor_queue = asyncio.Queue()
        reflector_queue = asyncio.Queue()
        completion_event = asyncio.Event()
        
        shared_state = {
            "final_answer": "",
            "previous_results": "",
            "iterations": 0,
            "current_task": None
        }

        # --- Worker 1: Planner ---
        async def _planner_worker():
            try:
                while not completion_event.is_set():
                    msg = await planner_queue.get()
                    
                    if shared_state["iterations"] >= max_iterations:
                        shared_state["final_answer"] = "Maximum iterations reached. Forcing completion."
                        completion_event.set()
                        planner_queue.task_done()
                        break

                    if shared_state.get("current_task") is None:
                        task = await self.planner.create_task(objective, prompt)
                        shared_state["current_task"] = task
                        await actor_queue.put("trigger_actor")
                    else:
                        shared_state["final_answer"] = shared_state["previous_results"] or "Task completed."
                        completion_event.set()
                        
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
                    msg = await actor_queue.get()
                    
                    task = shared_state.get("current_task")
                    if task is None:
                        actor_queue.task_done()
                        continue
                        
                    actor_output = await self.actor.generate_and_execute(
                        task, 
                        previous_results=shared_state["previous_results"]
                    )
                    
                    await reflector_queue.put((task, actor_output))
                    actor_queue.task_done()
            except asyncio.CancelledError:
                pass

        # --- Worker 3: Reflector ---
        async def _reflector_worker():
            try:
                while not completion_event.is_set():
                    task, actor_output = await reflector_queue.get()
                    
                    if actor_output.result and isinstance(actor_output.result, dict) and actor_output.result.get("status") == "finished":
                        await planner_queue.put("trigger_planner")
                        reflector_queue.task_done()
                        continue
                    
                    ref_input = ReflectorInputSchema(
                        reflector_type=ReflectorType.TASK,
                        target_id=actor_output.task_id,
                        target_content={"action_result": actor_output.dict()},
                        context=objective
                    )
                    
                    reflection = await self.reflector.reflect(ref_input)
                    
                    result_summary = f"\nAction Result: {actor_output.success}\nDetails: {actor_output.result}\nFeedback: {reflection.feedback}\nRating: {reflection.rating}/10\n"
                    shared_state["previous_results"] += result_summary
                    shared_state["iterations"] += 1
                    
                    self._schedule_background(memory.add_interaction(session_id, "system", f"Tool Output: {actor_output.success}"))
                    self._schedule_background(memory.long_term.add(session_id, reflection.feedback))
                    
                    await actor_queue.put("trigger_actor")
                    reflector_queue.task_done()
            except asyncio.CancelledError:
                pass

        # Spin up parallel tasks
        planner_task = asyncio.create_task(_planner_worker())
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

    async def run_stream(self, prompt: str, memory, session_id: str, max_iterations: int = 15):
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
            "iterations": 0,
            "current_task": None
        }

        async def _planner_worker():
            try:
                while not completion_event.is_set():
                    msg = await planner_queue.get()
                    
                    if shared_state["iterations"] >= max_iterations:
                        shared_state["final_answer"] = "Maximum iterations reached."
                        completion_event.set()
                        planner_queue.task_done()
                        break

                    if shared_state.get("current_task") is None:
                        await stream_queue.put({"type": "status", "content": f"🤔 Planner constructing architecture..."})
                        task = await self.planner.create_task(objective, prompt)
                        shared_state["current_task"] = task
                        await stream_queue.put({"type": "status", "content": "🛠️ Dispatching task to Actor Engine..."})
                        await actor_queue.put("trigger_actor")
                    else:
                        shared_state["final_answer"] = shared_state["previous_results"] or "Done."
                        completion_event.set()
                        
                    planner_queue.task_done()
            except asyncio.CancelledError:
                pass
            except Exception as e:
                shared_state["final_answer"] = f"Planner Fault: {e}"
                completion_event.set()

        async def _actor_worker():
            try:
                while not completion_event.is_set():
                    msg = await actor_queue.get()
                    
                    task = shared_state.get("current_task")
                    if task is None:
                        actor_queue.task_done()
                        continue
                        
                    actor_output = await self.actor.generate_and_execute(
                        task, 
                        previous_results=shared_state["previous_results"]
                    )
                    
                    if actor_output.result and isinstance(actor_output.result, dict) and actor_output.result.get("status") != "finished":
                        await stream_queue.put({"type": "status", "content": f"⚙️ Executed tool step. Success: {actor_output.success}"})
                        
                    await reflector_queue.put((task, actor_output))
                    actor_queue.task_done()
            except asyncio.CancelledError:
                pass

        async def _reflector_worker():
            try:
                while not completion_event.is_set():
                    task, actor_output = await reflector_queue.get()
                    
                    if actor_output.result and isinstance(actor_output.result, dict) and actor_output.result.get("status") == "finished":
                        await planner_queue.put("trigger_planner")
                        reflector_queue.task_done()
                        continue
                        
                    await stream_queue.put({"type": "status", "content": f"🧐 Reflecting on execution..."})
                    
                    ref_input = ReflectorInputSchema(
                        reflector_type=ReflectorType.TASK,
                        target_id=actor_output.task_id,
                        target_content={"action_result": actor_output.dict()},
                        context=objective
                    )
                    
                    reflection = await self.reflector.reflect(ref_input)
                    shared_state["previous_results"] += f"\nAction Result: {actor_output.success}\nDetails: {actor_output.result}\nFeedback: {reflection.feedback}\nRating: {reflection.rating}/10\n"
                    shared_state["iterations"] += 1
                    
                    self._schedule_background(memory.add_interaction(session_id, "system", f"Output: {actor_output.success}"))
                    self._schedule_background(memory.long_term.add(session_id, reflection.feedback))
                    
                    await actor_queue.put("trigger_actor")
                    reflector_queue.task_done()
            except asyncio.CancelledError:
                pass

        workers = [
            asyncio.create_task(_planner_worker()),
            asyncio.create_task(_actor_worker()),
            asyncio.create_task(_actor_worker()),
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
