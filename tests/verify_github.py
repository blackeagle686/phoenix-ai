import asyncio
import os
from phoenix.Phoenix AI import init_phoenix, startup_phoenix, get_rag_pipeline

async def verify_github_ingestion():
    # 1. Initialize
    print("\n--- [1] Initializing Phoenix AI SDK ---")
    init_phoenix()
    await startup_phoenix()
    rag = get_rag_pipeline()

    # 2. Ingest a public repository
    # Using a small repo for testing if possible, or just the current sdk for demo
    repo_url = "https://github.com/blackeagle686/phoenix-ai.git"
    print(f"\n--- [2] Testing GitHub Ingestion: {repo_url} ---")
    try:
        await rag.ingest_github(repo_url, branch="main")
        print("[+] GitHub ingestion completed successfully.")
    except Exception as e:
        print(f"[!] GitHub ingestion failed: {e}")

    # 3. Query the indexed source code
    print("\n--- [3] Testing Semantic Search on Code ---")
    # Query about something specific in the SDK code
    query = "How is the OpenAILLM class implemented?"
    answer = await rag.query(query)
    print(f"Query: {query}")
    print(f"AI Answer: {answer[:300]}...")

    print("\n--- Verification Finished ---")

if __name__ == "__main__":
    asyncio.run(verify_github_ingestion())

