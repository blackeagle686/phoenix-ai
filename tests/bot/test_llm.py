import asyncio
import os
import sys

# Add project root to path
sys.path.insert(0, os.getcwd())

from phoenix.services.llm.openai import OpenAILLM
from phoenix.services.llm.local import LocalLLM
from phoenix.core.config import config

async def test_llm_flow():
    print("\n" + "="*50)
    print("PHOENIX AI: LLM SERVICE TEST")
    print("="*50)

    # 1. Test OpenAILLM (with mock fallback)
    print("\n[1/2] Initializing OpenAILLM...")
    openai = OpenAILLM()
    await openai.init()
    
    prompt = "What is the capital of France?"
    print(f"[*] Sending prompt: {prompt}")
    
    try:
        response = await openai.generate(prompt)
        print(f"[v] Success! Response: {response}")
    except Exception as e:
        print(f"[!] OpenAI LLM Error: {e}")

    # 2. Test LocalLLM (with mock fallback)
    print("\n[2/2] Initializing LocalLLM...")
    local = LocalLLM()
    # Force mock for test if no model exists
    await local.init()
    
    prompt = "Write a short poem about code."
    print(f"[*] Sending prompt: {prompt}")
    
    try:
        response = await local.generate(prompt)
        print(f"[v] Success! Response: {response}")
    except Exception as e:
        print(f"[!] Local LLM Error: {e}")

    print("\n" + "="*50)
    print("LLM SERVICE TEST COMPLETE")
    print("="*50 + "\n")

if __name__ == "__main__":
    asyncio.run(test_llm_flow())

