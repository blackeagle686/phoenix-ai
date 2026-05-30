import asyncio
import os
from phoenix import init_phoenix_full, get_vlm_pipeline

async def main():
    # 1. Initialize Phoenix AI SDK (Connections, Models, Lifecycle)
    await init_phoenix_full()
    
    # 2. Get the VLM Pipeline (Orchestrates VLM + RAG + Cache)
    vlm = get_vlm_pipeline()
    
    # Create a dummy image for testing if it doesn't exist
    test_image = "test_vision.jpg"
    if not os.path.exists(test_image):
        with open(test_image, "wb") as f:
            f.write(b"\x00" * 100) # Dummy bytes
            
    print("\n[*] Testing VLM with Caching and RAG...")
    
    # 3. Ask a question (This is the 3rd line of user code)
    # The pipeline handles:
    # - Redis caching (Fast retrieval if asked before)
    # - Vector DB retrieval (Injecting context if use_rag=True)
    # - Vision Model execution (Async OpenAI or Local Ollama)
    
    response = await vlm.ask(
        prompt="Describe the contents of this scientific diagram.", 
        image_path=test_image,
        use_rag=True
    )
    
    print(f"\n[+] VLM Response:\n{response}")
    
    # Verify Caching
    print("\n[*] Testing Cache (Second call should be instant)...")
    import time
    start = time.time()
    response_cached = await vlm.ask(
        prompt="Describe the contents of this scientific diagram.", 
        image_path=test_image,
        use_rag=True
    )
    end = time.time()
    print(f"[+] Cache Response Time: {end - start:.4f}s")
    
    # Cleanup
    if os.path.exists(test_image):
        os.remove(test_image)

if __name__ == "__main__":
    asyncio.run(main())

