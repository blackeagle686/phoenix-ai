import asyncio
import os
import sys
import json
from pprint import pprint

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from phoenix.framework.agent.core.agent import Agent
from phoenix.services.llm.openai import OpenAILLM
from phoenix.framework.agent.tools.io import FileReadTool, FileWriteTool, FileSearchTool, FileEditTool, FileAppendTool
from phoenix.framework.agent.tools.code import PythonAnalyzerTool, CommandExecutionTool
from phoenix.framework.agent.execution.tool_manager import ToolManager
from phoenix.framework.agent.memory.hybrid import HybridMemory
from phoenix.framework.agent.cognition.reflector.schema import ReflectorInputSchema, ReflectorType

from phoenix.framework.agent.runtime.restricted_python import RestrictedPythonRuntime

async def test_centralized_brain_step_by_step():
    print("="*80)
    print("🚀 Testing Centralized Brain Architecture - Step-by-Step Flow")
    print("="*80)
    
    # 1. Setup LLM
    print("\n[*] Initializing OpenAILLM (LongCat-2.0-Preview)...")
    llm = OpenAILLM(
        api_key="ak_2yp3Xw1Ny7ky2pF7er9x93ZO9jj6G",
        model="LongCat-2.0-Preview",
        base_url="https://api.longcat.chat/openai"
    )
    await llm.init()
    
    # 2. Register core Tools and Runtime
    tools = [
        FileReadTool(),
        FileWriteTool(),
        FileEditTool(),
        FileAppendTool(),
        FileSearchTool(),
        PythonAnalyzerTool(),
        CommandExecutionTool()
    ]
    
    runtime = RestrictedPythonRuntime()
    
    # 3. Initialize Agent to get all components
    agent = Agent(llm=llm, tools=tools, runtime=runtime)
    tool_manager = agent.tool_manager
    print(f"[*] Registered {len(tools)} tools.")
    
    # Get components
    thinker = agent.thinker
    planner = agent.planner
    actor = agent.actor
    reflector = agent.reflector
    memory = agent.memory
    session_id = "test_session_001"
    
    prompt = (
        f"create a simple python script in  "
        "that just prints 'Hello from Centralized Brain!'. "
        "Use your tools to write this file to disk."
    )
    
    print("\n" + "-"*60)
    print(f"[*] USER PROMPT: {prompt}")
    print("-" * 60)
    
    # --- Brain Step 1: Planning ---
    print("\n>>> BRAIN STEP 1: Thinker generating PlanSchema...")
    plan = await planner.generate_initial_plan(prompt, memory, session_id)
    
    print("\n[📋 PLAN CREATED]")
    print(f"Objective: {plan.objective}")
    for idx, t in enumerate(plan.tasks):
        print(f"  Task {idx+1}: {t.task_id} - {t.description} (Status: {t.status})")
    print(json.dumps(plan.dict(), indent=2))
    
    if not plan.tasks:
        print("[!] No tasks generated. Exiting.")
        return
        
    for task_idx, current_task in enumerate(plan.tasks):
        if task_idx > 1: # Limit to 2 tasks for testing to prevent infinite loops
            print("\n[!] Stopping after 2 tasks for safety.")
            break
            
        print(f"\n{'='*40}")
        print(f">>> BRAIN STEP 2: Thinker defining ProblemSchema for Task '{current_task.task_id}'...")
        print(f"{'='*40}")
        problems = await planner.define_task_problems(current_task)
        
        print("\n[🔍 PROBLEMS DEFINED]")
        print(json.dumps(problems.model_dump(), indent=2))
        
        # --- Brain Step 3: Actor Solutions & Execution ---
        print(f"\n>>> BRAIN STEP 3: Thinker creating SolutionSchema & ActionSchema, Actor executing...")
        
        print("  -> Thinker generating SolutionSchema...")
        solution = await thinker.create_solutions(problems)
        print("\n[💡 SOLUTIONS CREATED]")
        print(json.dumps(solution.model_dump(), indent=2))
        
        print("\n  -> Thinker generating ActionSchema...")
        action_payload = await thinker.generate_action_payload(solution)
        print("\n[⚙️ ACTIONS TO TAKE]")
        print(json.dumps(action_payload.model_dump(), indent=2))
        
        # Execute tools via strict Runtime using Actor
        print("\n[🚀 EXECUTING ACTIONS VIA STRICT RUNTIME]")
        actor_output = await actor.generate_and_execute(problems)
        
        print("\n[✅ ACTOR EXECUTION RESULTS]")
        print(json.dumps(actor_output.model_dump(), indent=2))
        
        # --- Brain Step 4: Reflection ---
        print(f"\n>>> BRAIN STEP 4: Thinker generating ReflectionSchema to judge execution...")
        ref_input = ReflectorInputSchema(
            reflector_type=ReflectorType.TASK,
            target_id=current_task.task_id,
            target_content=actor_output.model_dump(),
            context=plan.objective
        )
        
        reflection = await reflector.reflect(ref_input)
        print("\n[🧐 REFLECTION CREATED]")
        print(json.dumps(reflection.model_dump(), indent=2))
        
        print("\n" + "="*80)
        if reflection.is_task_complete:
            print(f"✅ WORKFLOW COMPLETE: Task {current_task.task_id} was judged as successfully finished!")
        else:
            print(f"🔄 WORKFLOW INCOMPLETE: Task {current_task.task_id} requires more iterations.")
        print("="*80)

if __name__ == "__main__":
    asyncio.run(test_centralized_brain_step_by_step())
