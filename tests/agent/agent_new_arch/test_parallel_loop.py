import asyncio
import os
import sys

# Add the project root to sys.path (insert at index 0 to override installed packages)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from phoenix.framework.agent.core.agent import Agent
from phoenix.services.llm.openai import OpenAILLM
from phoenix.framework.agent.tools.io import FileReadTool, FileWriteTool, FileSearchTool
from phoenix.framework.agent.tools.code import PythonAnalyzerTool, CommandExecutionTool

async def test_parallel_arch():
    print("="*60)
    print("🚀 Testing Phoenix Parallel Agent Architecture")
    print("="*60)
    
    # 1. Setup LLM strictly using the requested configuration
    print("[*] Initializing OpenAILLM (LongCat-2.0-Preview)...")
    llm = OpenAILLM(
        api_key="ak_2yp3Xw1Ny7ky2pF7er9x93ZO9jj6G",
        model="LongCat-2.0-Preview",
        base_url="https://api.longcat.chat/openai"
    )
    
    # 2. Register core Tools
    tools = [
        FileReadTool(),
        FileWriteTool(),
        FileSearchTool(),
        PythonAnalyzerTool(),
        CommandExecutionTool()
    ]
    
    # 3. Initialize Agent
    print("[*] Instantiating Agent (Planner <-> Actor <-> Reflector)...")
    agent = Agent(
        llm=llm,
        tools=tools
    )
    
    # 4. Standard Run Test (File Generation)
    test_file = "test_parallel_output.txt"
    if os.path.exists(test_file):
        os.remove(test_file)
        
    prompt = (
        f" the create the project in this folder tests/agent/agent_new_arch/front_test"
    )
    
    print("\n" + "-"*40)
    print(f"[*] Phase 1: Running Standard Parallel Loop")
    print(f"[*] Prompt: {prompt}")
    print("-" * 40)
    
    result = await agent.run(prompt, max_iterations=6, mode="plan")
    
    print("\n[🎯] FINAL ANSWER:")
    print(result)
    
    # Verification
    if os.path.exists(test_file):
        print(f"\n[✅] SUCCESS! The file '{test_file}' was successfully created by the Agent.")
        with open(test_file, 'r') as f:
            print(f"[📝] File contents:\n{f.read().strip()}")
        # Cleanup
        os.remove(test_file)
    else:
        print(f"\n[❌] FAILED! The file '{test_file}' was NOT created.")
        
    # 5. Stream Run Test (Introspection)
    stream_prompt = (
        "Search the 'phoenix/framework/agent/cognition/actor/actor.py' file for the string 'ReflectorInputSchema' "
        "and provide a very brief summary of what it does."
    )
    
    print("\n" + "-"*40)
    print(f"[*] Phase 2: Running Streaming Parallel Loop")
    print(f"[*] Prompt: {stream_prompt}")
    print("-" * 40)
    
    print("\n[🌊] Streaming Output Started:\n")
    async for event in agent.run_stream(stream_prompt, max_iterations=6, mode="plan"):
        if event["type"] == "status":
            print(f"\n[STATUS] {event['content']}")
        elif event["type"] == "chunk":
            print(event["content"], end="", flush=True)
            
    print("\n\n" + "="*60)
    print("✅ Test Suite Complete")
    print("="*60)

if __name__ == "__main__":
    # Ensure graceful async exit
    try:
        asyncio.run(test_parallel_arch())
    except KeyboardInterrupt:
        print("\n[!] Test interrupted by user.")
