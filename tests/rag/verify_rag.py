import asyncio
import os
from phoenix.Phoenix AI import init_phoenix, get_rag_pipeline

async def verify_rag():
    print("Initializing Phoenix AI...")
    init_phoenix()
    
    rag = get_rag_pipeline()
    
    # 1. Prepare sample data
    test_dir = "./test_docs"
    os.makedirs(test_dir, exist_ok=True)
    with open(os.path.join(test_dir, "ai_info.txt"), "w") as f:
        f.write("Phoenix AI SDK is a powerful tool for building AI applications with RAG capabilities.\n")
        f.write("It supports ChromaDB and Qdrant as vector databases.\n")
    
    # 2. Ingest data
    print("Ingesting data...")
    await rag.ingest(test_dir)
    print("Ingestion complete.")
    
    # 3. Query data
    question = "What vector databases does Phoenix AI SDK support?"
    print(f"Querying: {question}")
    response = await rag.query(question)
    print(f"Response: {response}")
    
    # 4. Cleanup
    print("Cleaning up...")
    await rag.clear_data()
    import shutil
    shutil.rmtree(test_dir)
    if os.path.exists("./chroma_db"):
        shutil.rmtree("./chroma_db")
    print("Verification finished.")

if __name__ == "__main__":
    asyncio.run(verify_rag())

