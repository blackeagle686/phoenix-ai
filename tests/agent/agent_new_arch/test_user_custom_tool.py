import asyncio
import os
import sys

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from phoenix.framework.agent.core.agent import Agent
from phoenix.services.llm.openai import OpenAILLM
from phoenix.framework.agent.tools.base import tool

# 1. Define custom tool using the @tool decorator
@tool("say_hello", "Returns a greeting message for a given name. Input: 'name' (str).")
def say_hello(name: str = "World") -> str:
    return f"Hello, {name}! This is a custom tool executing successfully."

async def test_custom_tool():
    print("="*60)
    print("🚀 Testing Custom Tool with Phoenix Agent")
    print("="*60)
    
    # 2. Setup LLM strictly using the requested configuration
    llm = OpenAILLM(
        api_key="ak_2yp3Xw1Ny7ky2pF7er9x93ZO9jj6G",
        model="LongCat-2.0-Preview",
        base_url="https://api.longcat.chat/openai"
    )
    
    # 3. Pass the custom tool instance to the Agent
    tools = [say_hello]
    
    print("[*] Instantiating Agent with custom tool...")
    agent = Agent(llm=llm, tools=tools)
    
    prompt = "Please use your say_hello tool to greet 'Phoenix Developer'."
    
    print(f"\n[*] Prompt: {prompt}")
    print("-" * 40)
    
    result = await agent.run(prompt, max_iterations=4, mode="plan")
    
    print("\n[🎯] FINAL ANSWER:")
    print(result)

if __name__ == "__main__":
    # Ensure graceful async exit
    try:
        asyncio.run(test_custom_tool())
    except KeyboardInterrupt:
        print("\n[!] Test interrupted by user.")