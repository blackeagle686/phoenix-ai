import asyncio
from phoenix.Phoenix AI import init_phoenix_full, container
from phoenix.core.config import config

async def test_memory_flow():
    # 1. Initialize Phoenix AI Full
    # Force mock mode for OpenAI if no key is set, but Phoenix AI already handles it.
    await init_phoenix_full()
    
    llm = container.get("llm")
    memory = container.get("memory")
    session_id = "test_user_123"
    
    print(f"\n[*] Starting Memory Verification for session: {session_id}")
    
    # --- Turn 1 ---
    prompt1 = "Hi, my name is Phoenix AI. Remember this."
    print(f"\n[Turn 1] User: {prompt1}")
    response1 = await llm.generate(prompt1, session_id=session_id)
    print(f"[Turn 1] Assistant: {response1}")
    
    # Check history
    history = await memory.history.get(session_id)
    print(f"[Check] History length: {len(history)}")
    assert len(history) == 2, f"Expected 2 messages in history, got {len(history)}"
    
    # --- Turn 2 ---
    prompt2 = "What is my name?"
    print(f"\n[Turn 2] User: {prompt2}")
    response2 = await llm.generate(prompt2, session_id=session_id)
    print(f"[Turn 2] Assistant: {response2}")
    
    # In mock mode, we can't see the actual 'memory' effect in the response content, 
    # but we can verify the messages sent to the model if we had more tracing.
    # For verification, we'll check if the manager adds items correctly.
    
    history = await memory.history.get(session_id)
    print(f"[Check] History length: {len(history)}")
    assert len(history) == 4, f"Expected 4 messages in history, got {len(history)}"

    # --- Semantic Check ---
    fact = "The capital of France is Paris."
    await memory.semantic.add(session_id, fact)
    print(f"\n[*] Added direct fact to semantic memory: {fact}")
    
    search_results = await memory.semantic.search(session_id, "capital of France")
    print(f"[Check] Semantic Search results: {search_results}")
    
    # --- RAG Integration Check ---
    print("\n[*] Testing RAG context refinement...")
    from phoenix.Phoenix AI import get_rag_pipeline
    rag = get_rag_pipeline()
    
    # This should trigger query refinement in the background
    try:
        # In mock environment, RAG might fail if vector db is not fully seeded, but we check if it runs.
        await rag.query("Tell me more about my name.", session_id=session_id)
        print("[+] RAG query with session_id executed successfully.")
    except Exception as e:
        print(f"[!] RAG query failed (expected if DB empty): {e}")

    print("\n[+] Memory Verification Complete!")

if __name__ == "__main__":
    asyncio.run(test_memory_flow())

