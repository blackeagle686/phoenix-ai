import asyncio
import os
import sys

# Add the project root to sys.path (insert at index 0 to override installed packages)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from phoenix.framework.agent.core.agent import Agent
from phoenix.services.llm.openai import OpenAILLM
from phoenix.framework.agent.tools.io import FileReadTool, FileWriteTool, FileSearchTool, FileEditTool, FileAppendTool
from phoenix.framework.agent.tools.code import PythonAnalyzerTool, CommandExecutionTool

async def test_file_read():
    print("="*60)
    print("🚀 Testing Phoenix Parallel Agent Architecture: File Read")
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
        FileEditTool(),
        FileAppendTool(),
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
    test_file = "summary__AA.txt"
    if os.path.exists(test_file):
        os.remove(test_file)
        
    prompt = (
        f"Please read the file 'tests/agent/test_custom_user_model.py' and "
        f"write a very brief summary of what the file does into a new file called '{test_file}'."
    )
    
    print("\n" + "-"*40)
    print(f"[*] Phase 1: Running Standard Parallel Loop")
    print(f"[*] Prompt: {prompt}")
    print("-" * 40)
    
    print("\n[🌊] Streaming Output Started:\n")
    async for event in agent.run_stream(prompt, max_iterations=6, mode="plan"):
        if event["type"] == "status":
            print(f"\n[STATUS] {event['content']}")
        elif event["type"] == "chunk":
            print(event["content"], end="", flush=True)
    
    # Verification
    if os.path.exists(test_file):
        print(f"\n[✅] SUCCESS! The file '{test_file}' was successfully created by the Agent.")
        with open(test_file, 'r') as f:
            print(f"[📝] File contents:\n{f.read().strip()}")
        # Cleanup
        os.remove(test_file)
    else:
        print(f"\n[❌] FAILED! The file '{test_file}' was NOT created.")
        
    print("\n\n" + "="*60)
    print("✅ Test Suite Complete")
    print("="*60)

if __name__ == "__main__":
    # Ensure graceful async exit
    try:
        asyncio.run(test_file_read())
    except KeyboardInterrupt:
        print("\n[!] Test interrupted by user.")
