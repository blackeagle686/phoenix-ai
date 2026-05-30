import asyncio
import os
import sys
import json

# Add project root to path
sys.path.insert(0, os.getcwd())

from phoenix.main import init_phoenix, startup_phoenix, get_rag_pipeline

async def test_rag_sources_flow():
    print("\n" + "="*50)
    print("PHOENIX AI: RAG DATA SOURCES TEST")
    print("="*50)

    # 1. Initialization
    print("\n[1/5] Initializing Phoenix Services...")
    init_phoenix(local=False) 
    await startup_phoenix()
    rag = get_rag_pipeline()

    # 2. Test Local File Ingestion (Multiple Formats)
    print("\n[2/5] Testing Multi-Format File Ingestion...")
    test_dir = "tests/rag_data"
    os.makedirs(test_dir, exist_ok=True)
    
    # Create sample files
    files = {
        "doc.txt": "This is a plain text document about Phoenix AI.",
        "data.json": json.dumps({"project": "Phoenix AI", "version": "1.0.0", "description": "Agentic SDK"}),
        "table.csv": "name,value\nPhoenix,100\nAI,200"
    }
    
    for name, content in files.items():
        path = os.path.join(test_dir, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"[*] Created: {path}")

    print("[*] Ingesting directory...")
    await rag.ingest(test_dir)
    print("[v] Local directory ingestion complete.")

    # 3. Test URL Ingestion
    print("\n[3/5] Testing Web URL Ingestion...")
    test_url = "https://pypi.org/project/phoenix-ai/" # Just a placeholder
    print(f"[*] Scraping {test_url}...")
    try:
        await rag.ingest_url(test_url)
        print("[v] URL ingestion complete.")
    except Exception as e:
        print(f"[!] URL Ingestion failed: {e}")

    # 4. Test API Ingestion
    print("\n[4/5] Testing API Ingestion...")
    api_url = "https://api.github.com/repos/blackeagle686/phoenix-ai"
    print(f"[*] Fetching from API: {api_url}...")
    try:
        await rag.ingest_api(api_url)
        print("[v] API ingestion complete.")
    except Exception as e:
        print(f"[!] API Ingestion failed: {e}")

    # 5. Test GitHub Ingestion (Optional/Short)
    print("\n[5/5] Testing GitHub Repo Ingestion...")
    repo_url = "https://github.com/blackeagle686/phoenix-ai"
    print(f"[*] Cloning {repo_url}...")
    try:
        # We'll skip actual clone in some environments to save time, 
        # but the method is tested here.
        await rag.ingest_github(repo_url, branch="main")
        print("[v] GitHub ingestion complete.")
    except Exception as e:
        print(f"[!] GitHub Ingestion failed: {e}")

    # 6. Query the Knowledge Base
    print("\n[6/6] Querying Knowledge Base about Phoenix AI...")
    question = "Provide a detailed technical overview of Phoenix AI, including its core services and how to initialize it."
    print(f"[*] Question: {question}")
    
    try:
        response = await rag.query(question)
        print("\n" + "="*30 + " RAG RESPONSE " + "="*30)
        print(response)
        print("="*74)
    except Exception as e:
        print(f"[!] Query failed: {e}")

    # Cleanup local files
    print("\n[*] Cleaning up local test files...")
    import shutil
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    print("[v] Done.")

    print("\n" + "="*50)
    print("RAG DATA SOURCES TEST COMPLETE")
    print("="*50 + "\n")


if __name__ == "__main__":
    asyncio.run(test_rag_sources_flow())

