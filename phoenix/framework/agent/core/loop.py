from phoenix.framework.agent.cognition.thinker import Thinker
from phoenix.framework.agent.cognition.planner import Planner
from phoenix.framework.agent.cognition.reflector import Reflector
from phoenix.framework.agent.cognition.analyzer import Analyzer
from phoenix.framework.agent.cognition.actor import Actor
from phoenix.framework.agent.cognition.utils import generate_unique_id
from phoenix.framework.agent.cognition.pipeline import (
    BrainRegistry,
    CognitionPipeline,
    PipelineValidationError,
)
import asyncio
from pathlib import Path

class AgentLoop:
    """
    Coordinates the autonomous workflow:
    Think -> Analyze -> Plan -> Act -> Reflect
    """
    def __init__(self, thinker: Thinker, planner: Planner, actor: Actor, reflector: Reflector, analyzer: Analyzer):
        self.thinker = thinker
        self.planner = planner
        self.actor = actor
        self.reflector = reflector
        self.analyzer = analyzer
        self._background_tasks = set()
        self._registry = BrainRegistry()
        self._bootstrap_pipeline = self._load_pipeline_spec("default_bootstrap_pipeline.json")
        self._iteration_pipeline = self._load_pipeline_spec("default_iteration_pipeline.json")
        self._register_default_brains()

    def _load_pipeline_spec(self, filename: str) -> CognitionPipeline:
        spec_path = Path(__file__).resolve().parents[1] / "cognition" / "pipelines" / filename
        return CognitionPipeline.from_json_file(str(spec_path))

    def _register_default_brains(self):
        async def thinker_handler(data):
            objective = await self.thinker.analyze(data["prompt"], data["memory"], data["session_id"])
            return {"objective": objective}

        async def analyzer_handler(data):
            root_dir = data.get("root_dir", ".")
            analysis = await self.analyzer.analyze_workspace(data["prompt"], root_dir=root_dir)
            return {"analysis": analysis}

        async def planner_handler(data):
            plan = await self.planner.plan(
                data["objective"],
                task_file_id=data.get("task_file_id"),
                previous_results=data.get("previous_results", "")
            )
            return {"plan": plan}

        async def actor_handler(data):
            plan = data["plan"]["plan"] if "plan" in data["plan"] else data["plan"]
            action_result = await self.actor.execute(plan)
            actions = plan.get("actions", [])
            if not actions and "tool" in plan:
                actions = [{"tool": plan["tool"]}]
            actions_executed = len([a for a in actions if a.get("tool") != "finish"])
            has_finish = any(a.get("tool") == "finish" for a in actions)
            return {
                "action_result": action_result,
                "actions": actions,
                "actions_executed": actions_executed,
                "has_finish": has_finish
            }

        async def reflector_handler(data):
            action = data["action"]["plan"] if "plan" in data["action"] else data["action"]
            result_obj = data["result"]
            result_text = result_obj["action_result"] if isinstance(result_obj, dict) else str(result_obj)
            reflection = await self.reflector.reflect(data["objective"], action, result_text)
            return {"reflection": reflection}

        self._registry.register("thinker", thinker_handler)
        self._registry.register("analyzer", analyzer_handler)
        self._registry.register("planner", planner_handler)
        self._registry.register("actor", actor_handler)
        self._registry.register("reflector", reflector_handler)

    def register_brain(self, name: str, handler):
        """
        Public extension point: register custom cognition brain handler.
        Handler must be async and return Dict[str, Any].
        """
        self._registry.register(name, handler)

    def set_pipeline_specs(self, bootstrap_pipeline_path: str = None, iteration_pipeline_path: str = None):
        """
        Replace default JSON pipelines with user-provided specs.
        """
        if bootstrap_pipeline_path:
            self._bootstrap_pipeline = CognitionPipeline.from_json_file(bootstrap_pipeline_path)
        if iteration_pipeline_path:
            self._iteration_pipeline = CognitionPipeline.from_json_file(iteration_pipeline_path)

    def _schedule_background(self, coro):
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)

        def _on_done(t):
            self._background_tasks.discard(t)
            try:
                _ = t.exception()
            except Exception:
                pass

        task.add_done_callback(_on_done)

    async def run(self, prompt: str, memory, session_id: str, max_iterations: int = 5) -> str:
        """
        Executes the cognitive loop.
        Workflow: prompt -> Thinker(STM) -> Analyzer(Cache) -> Planner(Task Update) -> Actor -> Reflector(LTM)
        """
        # Bootstrap pipeline: Thinker -> Analyzer
        bootstrap_state = await self._bootstrap_pipeline.run(
            self._registry,
            {
                "prompt": prompt,
                "memory": memory,
                "session_id": session_id,
                "root_dir": "."
            }
        )
        objective = bootstrap_state["thinker_output"]["objective"]
        analysis = bootstrap_state["analyzer_output"]["analysis"]
        
        # Initialize task tracking
        task_file_id = generate_unique_id()
        memory.session.set("current_objective", objective)
        memory.session.set("project_analysis", analysis)
        memory.session.set("task_file_id", task_file_id)
        
        previous_results = ""
        final_answer = ""
        actions_taken = 0

        for i in range(max_iterations):
            # Iteration pipeline: Planner -> Actor -> Reflector
            iteration_state = await self._iteration_pipeline.run(
                self._registry,
                {
                    "objective": objective,
                    "task_file_id": task_file_id,
                    "previous_results": previous_results
                }
            )
            plan = iteration_state["planner_output"]["plan"]
            actor_output = iteration_state["actor_output"]
            action_result = actor_output["action_result"]
            reflection = iteration_state["reflector_output"]["reflection"]

            if actor_output["has_finish"]:
                if actions_taken == 0:
                    previous_results += "\nPlanner attempted to finish without actions. Validating...\n"
                    continue
                final_answer = previous_results or "Task completed successfully."
                break

            actions_taken += actor_output["actions_executed"]
            
            # Save to Memory (Short-Term + Long-Term)
            memory.reflection.add_reflection(reflection["reflection"])
            
            async def memory_updates():
                await asyncio.gather(
                    memory.add_interaction(session_id, "system", f"Action Result: {action_result}"),
                    # Step 6: Reflector adds to Long-Term Memory
                    memory.long_term.add(session_id, f"Learned from {objective}: {reflection['reflection']}"),
                    memory.consolidate_reflections(self.reflector.llm)
                )

            self._schedule_background(memory_updates())
            
            previous_results += f"\nAction: {plan}\nResult: {action_result}\nReflection: {reflection['reflection']}\n"
            
            if reflection["is_complete"]:
                final_answer = action_result
                break

        if not final_answer:
            final_answer = "Maximum iterations reached without full completion."

        await memory.add_interaction(session_id, "assistant", final_answer)
        return final_answer

    async def run_stream(self, prompt: str, memory, session_id: str, max_iterations: int = 5):
        """
        Streaming version of the workflow.
        """
        yield {"type": "status", "content": "🤔 Thinking & Analyzing..."}
        bootstrap_state = await self._bootstrap_pipeline.run(
            self._registry,
            {
                "prompt": prompt,
                "memory": memory,
                "session_id": session_id,
                "root_dir": "."
            }
        )
        objective = bootstrap_state["thinker_output"]["objective"]
        analysis = bootstrap_state["analyzer_output"]["analysis"]

        task_file_id = generate_unique_id()
        memory.session.set("current_objective", objective)
        memory.session.set("project_analysis", analysis)
        
        previous_results = ""
        final_answer = ""
        actions_taken = 0

        for i in range(max_iterations):
            yield {"type": "status", "content": f"🤔 Planner thinking (step {i + 1})..."}
            
            # Stream the thinking part of the planner
            async for thought in self.planner.stream_thinking(objective, task_file_id=task_file_id, previous_results=previous_results):
                yield {"type": "chunk", "content": thought}

            iteration_state = await self._iteration_pipeline.run(
                self._registry,
                {
                    "objective": objective,
                    "task_file_id": task_file_id,
                    "previous_results": previous_results
                }
            )
            plan = iteration_state["planner_output"]["plan"]
            actor_output = iteration_state["actor_output"]
            reflection = iteration_state["reflector_output"]["reflection"]
            action_result = actor_output["action_result"]

            if actor_output["has_finish"]:
                if actions_taken == 0:
                    previous_results += "\nWaiting for concrete tool results...\n"
                    continue
                final_answer = previous_results or "Done."
                break

            yield {"type": "status", "content": "🛠️ Executing actions..."}
            actions_taken += actor_output["actions_executed"]
            yield {"type": "status", "content": "🧐 Reflecting..."}
            memory.reflection.add_reflection(reflection["reflection"])
            
            async def memory_updates():
                await asyncio.gather(
                    memory.add_interaction(session_id, "system", f"Result: {action_result}"),
                    memory.long_term.add(session_id, reflection["reflection"]),
                    memory.consolidate_reflections(self.reflector.llm)
                )
            self._schedule_background(memory_updates())

            previous_results += f"\nResult: {action_result}\n"
            if reflection["is_complete"]:
                final_answer = action_result
                break

        yield {"type": "status", "content": "✅ Finalizing..."}
        await memory.add_interaction(session_id, "assistant", final_answer)
        for chunk in final_answer.split():
            yield {"type": "chunk", "content": chunk + " "}
