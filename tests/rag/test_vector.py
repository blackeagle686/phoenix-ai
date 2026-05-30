import asyncio
import os
import sys

# Add project root to path
sys.path.insert(0, os.getcwd())

from phoenix.services.vector.chroma import ChromaVectorDB

async def test_vector_flow():
    print("\n" + "="*50)
    print("PHOENIX AI: VECTOR DB SERVICE TEST")
    print("="*50)

    # 1. Initialization
    print("\n[1/3] Initializing ChromaDB (Persistent)...")
    vdb = ChromaVectorDB(collection_name="test_service_collection")
    await vdb.init()
    print("[v] Collection ready.")

    # 2. Add documents
    print("\n[2/3] Adding documents to Vector DB...")
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming the world.",
        "Python is a versatile programming language."
    ]
    metadatas = [{"type": "animal"}, {"type": "tech"}, {"type": "code"}]
    
    await vdb.add(texts=texts, metadatas=metadatas)
    print(f"[v] Success! Added {len(texts)} documents.")

    # 3. Search documents
    print("\n[3/3] Searching for 'programming'...")
    results = await vdb.search("programming", limit=2)
    
    if results:
        print(f"[v] Found {len(results)} matches:")
        for res in results:
            print(f"  - [{res['metadata']['type']}] {res['content']} (ID: {res['id'][:8]}...)")
    else:
        print("[!] No results found.")

    # Cleanup
    print("\n[*] Cleaning up test collection...")
    await vdb.clear()
    print("[v] Done.")

    print("\n" + "="*50)
    print("VECTOR DB SERVICE TEST COMPLETE")
    print("="*50 + "\n")

if __name__ == "__main__":
    asyncio.run(test_vector_flow())

