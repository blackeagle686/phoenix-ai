import asyncio
import os
import sys

# Add project root to path
sys.path.insert(0, os.getcwd())

from phoenix.main import init_phoenix, startup_phoenix, get_rag_pipeline

async def test_rag_flow():
    print("\n" + "="*50)
    print("PHOENIX AI: RAG SERVICE TEST")
    print("="*50)

    # 1. Initialization
    print("\n[1/4] Initializing Phoenix Services...")
    init_phoenix(local=False) # Use remote settings
    await startup_phoenix()
    
    # 2. Get RAG Pipeline
    print("\n[2/4] Setting up RAG Pipeline...")
    rag = get_rag_pipeline()
    
    # 3. Ingest a sample file (create one first)
    print("\n[3/4] Ingesting Sample Data...")
    sample_file = "tests/sample_data.txt"
    with open(sample_file, "w") as f:
        f.write("Phoenix AI is an advanced agentic coding SDK.\n")
        f.write("It supports RAG, VLM, and memory-managed interactions.\n")
        f.write("The current version is 1.0.0.\n")
    
    print(f"[*] Created {sample_file}")
    await rag.ingest(sample_file)
    print("[v] Ingestion complete.")
    
    # 4. Query the RAG
    print("\n[4/4] Querying RAG Pipeline...")
    question = "What is the version of Phoenix AI?"
    print(f"[*] Question: {question}")
    
    try:
        # Since we are in mock/local mode without real LLM, it will return a mock response
        # or use the local LLM if configured.
        response = await rag.query(question)
        print(f"[v] Success! RAG Response: {response}")
    except Exception as e:
        print(f"[!] RAG Query Error: {e}")

    # Cleanup
    if os.path.exists(sample_file):
        os.remove(sample_file)

    print("\n" + "="*50)
    print("RAG SERVICE TEST COMPLETE")
    print("="*50 + "\n")

if __name__ == "__main__":
    asyncio.run(test_rag_flow())

