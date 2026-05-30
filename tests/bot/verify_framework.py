import asyncio
import os
import shutil
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from phoenix import ChatBot

async def verify_chatbot():
    print("--- [Phoenix AI Bot Framework Verification] ---")
    
    # 1. Setup sample data for RAG
    test_dir = "./test_framework_docs"
    os.makedirs(test_dir, exist_ok=True)
    with open(os.path.join(test_dir, "framework_info.txt"), "w") as f:
        f.write("Phoenix AI Framework allows building bots in one line.\n")
        f.write("It supports Builder pattern with fluent API.\n")

    try:
        # 2. Build the Bot (One-liner!)
        print("Building ChatBot...")
        bot = (ChatBot(local=False, vlm=True, tts=True, stt=True)
               .with_rag(data_to_insight_path=test_dir)
               .with_memory()
               .build())

        # 3. Test Text Interaction (with RAG and Memory)
        print("\nTesting Text Interaction (RAG + Memory)...")
        response = await bot.chat("How do I build bots in Phoenix AI?")
        print(f"Bot (Text): {response['text'] if isinstance(response, dict) else response}")

        # 4. Test Second Interaction (Memory Check)
        print("\nTesting Memory...")
        response2 = await bot.chat("What did I just ask you?")
        print(f"Bot (Memory): {response2['text'] if isinstance(response2, dict) else response2}")

        # 5. Test Audio Interaction (Mock STT)
        print("\nTesting Audio Input (STT)...")
        audio_response = await bot.chat(audio_path="dummy_voice.wav")
        print(f"Bot (STT Response): {audio_response['text'] if isinstance(audio_response, dict) else audio_response}")

        # 6. Test Vision Interaction (Mock VLM)
        print("\nTesting Vision Input (VLM)...")
        with open("dummy_image.jpg", "w") as f: f.write("dummy")
        vision_response = await bot.chat("What is in this image?", image_path="dummy_image.jpg")
        print(f"Bot (Vision): {vision_response['text'] if isinstance(vision_response, dict) else vision_response}")
        os.remove("dummy_image.jpg")

    finally:
        # Cleanup
        print("\nCleaning up...")
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
        if os.path.exists("./chroma_db"):
            shutil.rmtree("./chroma_db")
        print("Verification complete.")

if __name__ == "__main__":
    asyncio.run(verify_chatbot())

