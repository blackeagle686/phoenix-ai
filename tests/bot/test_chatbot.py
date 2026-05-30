import asyncio
import os
import sys

# Add project root to path
sys.path.insert(0, os.getcwd())

from phoenix import ChatBot

async def test_chatbot_flow():
    print("\n" + "="*50)
    print("PHOENIX AI: CHATBOT BUILDER TEST")
    print("="*50)

    # 1. Build a simple bot
    print("\n[1/3] Building Simple ChatBot...")
    bot = ChatBot(local=True) \
        .with_system_prompt("You are a specialized math assistant.") \
        .with_security(mode="standard") \
        .build()
    
    print("[v] ChatBot instance built successfully.")

    # 2. Test interaction
    print("\n[2/3] Sending message to ChatBot...")
    question = "What is 2+2?"
    print(f"[*] User: {question}")
    
    try:
        response = await bot.chat(text=question)
        print(f"[v] Bot Response: {response}")
    except Exception as e:
        print(f"[!] ChatBot Error: {e}")

    # 3. Test multimodal builder
    print("\n[3/3] Building Multimodal ChatBot (Mock)...")
    try:
        mm_bot = ChatBot(local=True, vlm=True) \
            .with_memory() \
            .build()
        print("[v] Multimodal ChatBot built successfully.")
    except Exception as e:
        print(f"[!] Builder Error: {e}")

    print("\n" + "="*50)
    print("CHATBOT BUILDER TEST COMPLETE")
    print("="*50 + "\n")

if __name__ == "__main__":
    asyncio.run(test_chatbot_flow())

