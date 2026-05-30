import asyncio
import os
import sys

# Add the project root to sys.path so we can import 'phoenix'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from phoenix.framework.agent.core import Agent
from phoenix.services.llm.openai import OpenAILLM
from phoenix.framework.agent.memory.hybrid import HybridMemory
from phoenix.framework.agent.tools.registry import ToolRegistry

async def main():
    # 1. Initialize custom LLM (User asked to use OpenAILLM default key)
    # OpenAILLM uses LongCat-Flash-Chat and its default configuration when instantiated
    llm = OpenAILLM()
    
    # 2. Setup Hybrid Memory
    memory = HybridMemory()
    
    # 3. Load default tools (Web Search, Code Execution)
    tools = ToolRegistry.load_default()
    
    # 4. Instantiate Agent
    agent = Agent(llm=llm, memory=memory, tools=tools)
    
    # 5. Define and register a custom tool using the @tool decorator
    from phoenix.framework.agent.tools import tool
    
    @tool(name="custom_math", description="Calculates the square of a given number. Input: 'number' (int).")
    def custom_math_tool(number: int):
        return f"The square of {number} is {number ** 2}"
        
    agent.register_tool(custom_math_tool)
    
    print("Agent Initialized. Custom tools registered. Running task...")
    
    # 5. Run the Agent with a complex multi-step task
    task_prompt = (
        "Your task is to write a python script that solves a mathematical problem, save it, and test it. "
        "Step 1: Write a python function that calculates the 10th number in the Fibonacci sequence. "
        "Step 2: Save this code into a file named 'fibonacci_solver.py' using the python_repl tool. "
        "Step 3: Read the file back to verify its contents. "
        "Step 4: Execute the script or import it using the python_repl tool to get the 10th Fibonacci number and print the result. "
        "Step 5: Review the output and confirm if the sequence calculation was successful."
    )
    result = await agent.run(task_prompt, max_iterations=8)
    
    print("\n--- Final Output ---")
    print(result)
    
    print("\n--- Agent Reflections ---")
    print(memory.reflection.get_reflections())

if __name__ == "__main__":
    asyncio.run(main())

