import asyncio
import os
import sys

# Add the project root to sys.path (insert at index 0 to override installed packages)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from phoenix.framework.agent.core.agent import Agent
from phoenix.services.llm.openai import OpenAILLM
from phoenix.framework.agent.tools.io import FileReadTool, FileWriteTool, FileSearchTool, FileEditTool, FileAppendTool
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
        FileEditTool(),
        FileAppendTool(),
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
    
    # 4. Standard Run Test (Frontend Project Update)
    prompt = (
        "create a modular frontend wallet system project tests/agent/agent_new_arch/front_test "
        "The system should have the following features:"
        "- Wallet Creation (generate mnemonic phrase, generate private key)"
        "- Wallet Recovery (recover from mnemonic phrase)"
        "- Transaction Creation (send transaction to another wallet)"
        "- Transaction Broadcasting (broadcast transaction to blockchain)"
        "- Transaction History (list of transactions)"
        "- Balance Checking (check balance of wallet)"
    )
    
    print("\n" + "-"*40)
    print(f"[*] Phase 1: Running Standard Parallel Loop (Frontend Update Test)")
    print(f"[*] Prompt: {prompt}")
    print("-" * 40)
    
    # Increase max_iterations slightly since updating multiple files can take a few steps
    result = await agent.run(prompt, max_iterations=25, mode="plan")
    
    print("\n[🎯] FINAL ANSWER:")
    print(result)
    
    print("\n[✅] Execution completed! Please check tests/agent/agent_new_arch/front_test/ manually to verify the updates.")
        
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
