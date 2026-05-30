import asyncio
import os
import sys

# Add project root to path
sys.path.insert(0, os.getcwd())

from phoenix.main import init_phoenix, startup_phoenix, get_memory

async def test_memory_flow():
    print("\n" + "="*50)
    print("PHOENIX AI: MEMORY SERVICE TEST")
    print("="*50)

    # 1. Initialization
    print("\n[1/4] Initializing Memory Service...")
    init_phoenix(local=True)
    await startup_phoenix()
    
    memory = get_memory()
    session_id = "test_user_123"

    # 2. Add interactions
    print("\n[2/4] Adding Interactions to Memory...")
    interactions = [
        ("My favorite color is blue.", "That's nice! I'll remember that your favorite color is blue."),
        ("I live in Cairo.", "Understood. You are located in Cairo, Egypt.")
    ]
    
    for q, a in interactions:
        print(f"[*] Storing interaction: Q: {q}")
        await memory.add_interaction(session_id, q, a)
    
    print("[v] Interactions stored.")

    # 3. Retrieve context (History)
    print("\n[3/4] Retrieving Conversation History...")
    history = await memory.get_context(session_id)
    print(f"[v] Current History:\n{history}")

    # 4. Search memory (Semantic)
    print("\n[4/4] Searching Semantic Memory...")
    query = "What do you know about my preferences?"
    print(f"[*] Query: {query}")
    
    past_facts = await memory.search_memory(session_id, query)
    if past_facts:
        print(f"[v] Found relevant facts:\n{past_facts}")
    else:
        print("[ ] No relevant facts found in semantic memory (might require actual Vector DB search).")

    print("\n" + "="*50)
    print("MEMORY SERVICE TEST COMPLETE")
    print("="*50 + "\n")

if __name__ == "__main__":
    asyncio.run(test_memory_flow())

