import asyncio
import os
import shutil
from phoenix.framework.chatbot.core import ChatBot
from phoenix.core.config import config

async def test_chatbot_full_flow():
    print("\\n--- Testing ChatBot Full Flow ---")
    
    # 1. Setup dummy data for RAG
    rag_dir = "tests/test_021/dummy_rag_data"
    os.makedirs(rag_dir, exist_ok=True)
    with open(os.path.join(rag_dir, "info.txt"), "w") as f:
        f.write("The secret code for Phoenix is: PHX-2024-BETA. The capital of France is Paris.")

    try:
        # 2. Build ChatBot with specific LongCat config
        bot = (ChatBot(local=False)
               .with_openai(api_key="ak_2yp3Xw1Ny7ky2pF7er9x93ZO9jj6G", 
                            base_url="https://api.longcat.chat/openai")
               .with_model(llm="LongCat-Flash-Chat")
               .with_rag(rag_dir)
               .with_memory()
               .with_system_prompt("You are a helpful AI assistant for the Phoenix framework.")
               .with_security(mode="standard")
               .build())

        # 3. First Interaction (RAG + Memory)
        print("User: What is the secret code for Phoenix?")
        response1 = await bot.chat("What is the secret code for Phoenix?")
        print(f"Bot: {response1}")
        assert "PHX-2024-BETA" in response1, "RAG failed to retrieve the secret code."

        # 4. Second Interaction (Contextual Memory)
        print("User: What else did I ask you about?")
        response2 = await bot.chat("What else did I ask you about?")
        print(f"Bot: {response2}")
        assert "secret code" in response2.lower() or "phoenix" in response2.lower(), "Memory failed to track the conversation history."

        # 5. Modality Check (System Prompt)
        print("User: Who are you?")
        response3 = await bot.chat("Who are you?")
        print(f"Bot: {response3}")
        assert "assistant" in response3.lower() and "phoenix" in response3.lower(), "System prompt was ignored."

        print("\\n✅ ChatBot Full Flow Test Passed!")

    finally:
        # Cleanup
        if os.path.exists(rag_dir):
            shutil.rmtree(rag_dir)

if __name__ == "__main__":
    if not config.OPENAI_API_KEY and not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found. Please set it in your environment for full integration testing.")
    else:
        asyncio.run(test_chatbot_full_flow())

