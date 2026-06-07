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
        f"create a simple python script in {os.path.abspath(os.path.dirname(__file__))}/hello_world.py "
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
        
    # Take the first task
    current_task = plan.tasks[0]
    
    # --- Brain Step 2: Defining Problems ---
    print(f"\n>>> BRAIN STEP 2: Thinker defining ProblemSchema for Task '{current_task.task_id}'...")
    problems = await planner.define_task_problems(current_task)
    
    print("\n[🔍 PROBLEMS DEFINED]")
    print(json.dumps(problems.dict(), indent=2))
    
    # --- Brain Step 3: Actor Solutions & Execution ---
    print(f"\n>>> BRAIN STEP 3: Thinker creating SolutionSchema & ActionSchema, Actor executing...")
    
    # We will step into actor.generate_and_execute manually to print the schemas
    print("  -> Thinker generating SolutionSchema...")
    solution = await thinker.create_solutions(problems)
    print("\n[💡 SOLUTIONS CREATED]")
    print(json.dumps(solution.dict(), indent=2))
    
    print("\n  -> Thinker generating ActionSchema...")
    action_payload = await thinker.generate_action_payload(solution)
    print("\n[⚙️ ACTIONS TO TAKE]")
    print(json.dumps(action_payload.dict(), indent=2))
    
    # Execute tools
    print("\n[🚀 EXECUTING ACTIONS]")
    results = []
    success = True
    
    # IO Operations
    for io_op in action_payload.io_operations:
        print(f"  -> Executing IO Op: {io_op.operation} on {io_op.file_path}")
        if io_op.operation in ["create", "edit"]:
            try:
                with open(io_op.file_path, "w") as f:
                    f.write(io_op.content)
                res_dict = {"io_op": io_op.operation, "path": io_op.file_path, "success": True}
                print(f"     ✅ Written to disk successfully.")
            except Exception as e:
                res_dict = {"io_op": io_op.operation, "path": io_op.file_path, "success": False, "error": str(e)}
                print(f"     ❌ Failed: {e}")
                success = False
            results.append(res_dict)
            
    # Tools
    for tool_call in action_payload.tools_to_call:
        print(f"  -> Executing Tool: {tool_call.tool_name} with args: {tool_call.arguments}")
        try:
            res_str = await tool_manager.execute_tool(tool_call.tool_name, tool_call.arguments)
            res_success = "error" not in str(res_str).lower()
            results.append({"tool": tool_call.tool_name, "success": res_success, "output": str(res_str)})
            print(f"     ✅ Tool returned: {str(res_str)[:100]}...")
        except Exception as e:
            results.append({"tool": tool_call.tool_name, "success": False, "error": str(e)})
            print(f"     ❌ Tool error: {e}")
            success = False
            
    actor_output = {
        "task_id": current_task.task_id,
        "success": success,
        "execution_results": results,
        "action_plan": action_payload.action_plan
    }
    
    # --- Brain Step 4: Reflection ---
    print(f"\n>>> BRAIN STEP 4: Thinker generating ReflectionSchema to judge execution...")
    ref_input = ReflectorInputSchema(
        reflector_type=ReflectorType.TASK,
        target_id=current_task.task_id,
        target_content=actor_output,
        context=plan.objective
    )
    
    reflection = await reflector.reflect(ref_input)
    print("\n[🧐 REFLECTION CREATED]")
    print(json.dumps(reflection.dict(), indent=2))
    
    print("\n" + "="*80)
    if reflection.is_task_complete:
        print("✅ WORKFLOW COMPLETE: Task was judged as successfully finished!")
    else:
        print("🔄 WORKFLOW INCOMPLETE: Task requires more iterations.")
    print("="*80)

if __name__ == "__main__":
    asyncio.run(test_centralized_brain_step_by_step())
