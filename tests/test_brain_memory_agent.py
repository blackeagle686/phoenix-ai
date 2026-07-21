import asyncio
import os
import sys
import time

# Set up OpenRouter configs
os.environ["OPENAI_API_KEY"] = "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"
os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
os.environ["OPENAI_LLM_MODEL"] = "poolside/laguna-s-2.1:free"

# Add the project root to sys.path so we can import 'phoenix'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from phoenix.framework.agent.core.agent import Agent
from phoenix.services.llm.openai import OpenAILLM
from phoenix.framework.agent.tools.registry import ToolRegistry
from phoenix.framework.agent.memory.manager import MemoryManager

# Import the new Rust PyO3 library
import brain_memory

class BrainMemoryAdapter(MemoryManager):
    """
    A MemoryManager that bridges the Python short-term cell memory
    with the Rust Long-Term BrainMemory.
    """
    def __init__(self):
        super().__init__()
        self.brain = brain_memory.BrainMemoryClient()
        print("[BrainMemory] Rust PyO3 Extension Initialized")

    async def add_interaction(self, session_id: str, role: str, content: str, metadata: dict = None):
        # 1. Call the original short-term memory manager to keep AgentLoop working
        await super().add_interaction(session_id, role, content, metadata)
        
        # 2. Asynchronously send this as an Event to the Rust Backend
        import json
        event = {
            "event_type": "interaction",
            "perception_id": "none",
            "payload": {"role": role, "content": content},
            "embedding": [],
            "timestamp": "2026-07-21T00:00:00Z",
            "importance": 1.0,
            "id": f"{session_id}_{len(self.interactions)}"
        }
        
        # Benchmark the Rust call
        start_time = time.perf_counter()
        self.brain.add_event(json.dumps(event))
        duration = (time.perf_counter() - start_time) * 1000
        print(f"[BrainMemory] Rust backend absorbed Event in {duration:.3f}ms")

    async def get_full_context(self, session_id: str = "", query: str = "") -> str:
        # Benchmark the Rust Retrieval Pipeline
        start_time = time.perf_counter()
        wm_json = self.brain.retrieve_working_memory(query)
        duration = (time.perf_counter() - start_time) * 1000
        print(f"[BrainMemory] Rust backend built Working Memory in {duration:.3f}ms")
        
        # Fall back to base memory for the loop context, but we can prepend the Brain Knowledge
        base_context = await super().get_full_context(session_id, query)
        
        return f"--- KNOWLEDGE FROM BRAIN ---\n{wm_json}\n\n--- CURRENT SESSION ---\n{base_context}"

async def main():
    llm = OpenAILLM(
        api_key="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",
        base_url="https://openrouter.ai/api/v1",
        model="poolside/laguna-s-2.1:free"
    )
    memory = BrainMemoryAdapter()
    tools = ToolRegistry.load_default()
    
    agent = Agent(llm=llm, memory=memory, tools=tools)
    
    print("Agent Initialized with Rust BrainMemory backend. Running task...")
    
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

if __name__ == "__main__":
    asyncio.run(main())
