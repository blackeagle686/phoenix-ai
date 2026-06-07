import asyncio
from typing import Dict, Any, Optional

from phoenix.framework.agent.cognition.thinker import Thinker
from phoenix.framework.agent.cognition.planner import Planner
from phoenix.framework.agent.cognition.reflector import Reflector
from phoenix.framework.agent.cognition.analyzer import Analyzer
from phoenix.framework.agent.cognition.actor import Actor
from phoenix.framework.agent.cognition.reflector.schema import ReflectorInputSchema, ReflectorType

class AgentLoop:
    """
    Coordinates the autonomous workflow with a parallel Async Channel Architecture.
    Implements the Two-Phase Brain Step:
    1. Brain Step 1: Generate full plan based on prompt.
    2. Brain Step 2+: Iterate over tasks, solving them systematically.
    """
    def __init__(self, thinker: Thinker, planner: Planner, actor: Actor, reflector: Reflector, analyzer: Analyzer):
        self.thinker = thinker
        self.planner = planner
        self.actor = actor
        self.reflector = reflector
        self.analyzer = analyzer
        self._background_tasks = set()

    def _schedule_background(self, coro):
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def run(self, prompt: str, memory, session_id: str, max_iterations: int = 15) -> str:
        # --- BRAIN STEP 1: Planning Phase ---
        plan = await self.planner.generate_initial_plan(prompt, memory, session_id)
        memory.session.set("current_objective", plan.objective)
        
        # --- BRAIN STEP 2+: Execution Phase ---
        final_answer = ""
        iterations = 0
        
        for task in plan.tasks:
            if task.status == "done":
                continue
                
            task_complete = False
            
            # Sub-step: Define explicit problems for the task
            problems = await self.planner.define_task_problems(task)
            
            while not task_complete and iterations < max_iterations:
                # Ask Actor to solve the problems and execute
                actor_output = await self.actor.generate_and_execute(problems)
                
                # Ask Reflector to judge the runtime output
                ref_input = ReflectorInputSchema(
                    reflector_type=ReflectorType.TASK,
                    target_id=task.task_id,
                    target_content=actor_output.dict(),
                    context=plan.objective
                )
                
                reflection = await self.reflector.reflect(ref_input)
                
                # Update memory
                self._schedule_background(memory.add_interaction(session_id, "system", f"Executed tools for {task.task_id}. Success: {actor_output.success}"))
                self._schedule_background(memory.long_term.add(session_id, reflection.feedback))
                
                iterations += 1
                
                if reflection.is_task_complete:
                    task_complete = True
                    task.status = "done"
                    final_answer += f"Task {task.task_id} completed.\n"
                else:
                    # Provide feedback to the problem context for the next iteration
                    # In a fully fleshed out version, we would mutate `problems` with the feedback here.
                    pass
                    
        if iterations >= max_iterations:
            final_answer += "\nMax iterations reached before completing all tasks."
            
        await memory.add_interaction(session_id, "assistant", final_answer)
        return final_answer

    async def run_stream(self, prompt: str, memory, session_id: str, max_iterations: int = 15):
        # --- BRAIN STEP 1: Planning Phase ---
        yield {"type": "status", "content": "🤔 Brain Step 1: Thinking and Planning..."}
        plan = await self.planner.generate_initial_plan(prompt, memory, session_id)
        memory.session.set("current_objective", plan.objective)
        
        yield {"type": "status", "content": f"📋 Plan generated: {len(plan.tasks)} tasks."}
        
        # --- BRAIN STEP 2+: Execution Phase ---
        final_answer = ""
        iterations = 0
        
        for task in plan.tasks:
            if task.status == "done":
                continue
                
            task_complete = False
            
            yield {"type": "status", "content": f"🔍 Brain Step 2: Defining problems for Task {task.task_id}..."}
            problems = await self.planner.define_task_problems(task)
            
            while not task_complete and iterations < max_iterations:
                yield {"type": "status", "content": f"⚙️ Actor is generating solutions and executing..."}
                
                actor_output = await self.actor.generate_and_execute(problems)
                
                yield {"type": "status", "content": f"🧐 Reflector is judging the execution..."}
                ref_input = ReflectorInputSchema(
                    reflector_type=ReflectorType.TASK,
                    target_id=task.task_id,
                    target_content=actor_output.dict(),
                    context=plan.objective
                )
                
                reflection = await self.reflector.reflect(ref_input)
                
                self._schedule_background(memory.add_interaction(session_id, "system", f"Executed tools for {task.task_id}. Success: {actor_output.success}"))
                self._schedule_background(memory.long_term.add(session_id, reflection.feedback))
                
                iterations += 1
                
                if reflection.is_task_complete:
                    task_complete = True
                    task.status = "done"
                    final_answer += f"Task {task.task_id} completed. Rating: {reflection.rating}/10.\n"
                    yield {"type": "status", "content": f"✅ Task {task.task_id} marked as complete!"}
                else:
                    yield {"type": "status", "content": f"🔄 Task incomplete. Adjusting based on feedback..."}
                    
        if iterations >= max_iterations:
            final_answer += "\nMax iterations reached before completing all tasks."
            
        await memory.add_interaction(session_id, "assistant", final_answer)
        yield {"type": "status", "content": "✅ Brain loop finished."}
        yield {"type": "chunk", "content": final_answer}
