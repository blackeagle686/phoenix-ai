# Phoenix AI RAG Framework

The Phoenix AI SDK provides a suite of Retrieval-Augmented Generation (RAG) frameworks tailored for different architectural needs. Each framework handles ingestion, vector storage, chunking, retrieval optimization (HyDE, MMR, Compression), and LLM generation automatically.

All RAG systems share the same unified API for ingestion and querying. You can seamlessly swap between `RAG`, `CAG`, `AgenticRAG`, `AdaptiveRAG`, and `MultiModalRAG` based on your project requirements.

---

## 1. Standard / Advanced RAG

Best for direct Q&A against static documents, codebases, or knowledge bases. It features built-in advanced retrieval techniques like HyDE (Hypothetical Document Embeddings), query expansion, and MMR reranking.

```python
import asyncio
from phoenix.framework.rag import RAG, RAGConfig

async def main():
    # Initialize with advanced retrieval features
    rag = RAG(
        config=RAGConfig(
            chunk_size=500,
            hyde_enabled=True,
            query_expansion=True,
            mmr_enabled=True,
            context_compression=True
        )
    )

    # Ingest a local folder (supports PDF, DOCX, CSV, JSON, TXT, Code, etc.)
    await rag.ingest("./knowledge_base")

    # Query the knowledge base
    answer = await rag.query("What are the core features of the system?")
    print(answer)

    # Query and retrieve sources
    result = await rag.query_with_sources("Explain the architecture.")
    print("Answer:", result["answer"])
    print("Sources:", result["sources"])

asyncio.run(main())
```

---

## 2. Multi-Source Ingestion

The framework supports unified, declarative ingestion from almost any data source using `ingest_multi`.

```python
from phoenix.framework.rag import RAG

async def main():
    rag = RAG()

    # Ingest from multiple varied sources simultaneously
    await rag.ingest_multi([
        # Local Folder
        {"type": "folder", "path": "/data/internal_docs"},
        
        # GitHub Repository
        {"type": "github", "repo_url": "https://github.com/user/project", "branch": "main"},
        
        # SQL Database (ingest 'body' column from 'articles' table)
        {
            "type": "sql", 
            "connection_string": "postgresql://user:pass@localhost:5432/db", 
            "table": "articles", 
            "text_columns": "body"
        },
        
        # REST API (fetch data, paginate using 'next_url', use 'content' field)
        {
            "type": "api", 
            "url": "https://api.example.com/v1/posts", 
            "data_path": "results", 
            "text_field": "content",
            "pagination": {"next_field": "next_url", "max_pages": 5}
        },
        
        # Google Drive (requires credentials)
        {
            "type": "google_drive", 
            "folder_id": "1A2B3C4D5E6F7G8H9I0J", 
            "credentials_path": "./service_account.json"
        },
        
        # Batch URLs
        {"type": "urls", "urls": ["https://example.com/guide", "https://example.com/faq"]}
    ])

    answer = await rag.query("Summarize the latest API posts and internal guidelines.")
    print(answer)
```

---

## 3. Cache-Augmented Generation (CAG)

Best for high-traffic environments where users ask similar questions frequently. CAG uses a semantic cache (embeddings-based) to intercept queries and return cached answers instantly, saving LLM tokens and vector DB lookups.

```python
from phoenix.framework.rag import CAG, CAGConfig

async def main():
    cag = CAG(
        config=CAGConfig(
            semantic_cache_threshold=0.92, # 92% similarity triggers a cache hit
            cache_ttl=3600 # 1 hour
        )
    )

    await cag.ingest("./faq_docs")

    # Optional: Preload the cache with common questions
    await cag.preload_corpus([
        "How do I reset my password?",
        "What are the business hours?",
        "Where is the headquarters?"
    ])

    # First query (might take 2 seconds)
    ans1 = await cag.query("I forgot my password, how to reset?")
    
    # Second query (Instant hit! Same semantic meaning)
    ans2 = await cag.query("Can you help me recover my password?")
```

---

## 4. Agentic RAG

Transforms passive retrieval into an autonomous agent. Agentic RAG can verify its own answers, rewrite poor user queries, and call external tools when vector search isn't enough.

```python
from phoenix.framework.rag import AgenticRAG, AgenticRAGConfig

async def get_weather(location: str):
    # Mock tool
    return f"The weather in {location} is 72F and sunny."

async def main():
    arag = AgenticRAG(
        config=AgenticRAGConfig(
            max_retries=3,          # Retry up to 3 times
            verify_answer=True,     # Verify generated answer against context
            rewrite_on_fail=True,   # Rewrite query if context is irrelevant
            routing_enabled=True    # Route between vector DB, direct answers, and tools
        ),
        tools=[{"name": "get_weather", "fn": get_weather, "description": "Get current weather"}]
    )

    await arag.ingest("./knowledge_base")

    # Scenario 1: Will route to vector_search, verify context, and answer
    ans1 = await arag.query("What is the refund policy?")

    # Scenario 2: Will route to the 'get_weather' tool automatically
    ans2 = await arag.query("What is the weather like in New York?")

    # Debugging: Get the full reasoning trace
    result = await arag.query_with_trace("How do I install the software?")
    print("Answer:", result["answer"])
    print("Trace:", result["trace"]["steps"])
```

---

## 5. Adaptive RAG

Best for conversational chatbots. It maintains multi-turn session memory, resolves pronoun references (e.g., turning "what about it?" into "what about the refund policy?"), and dynamically adjusts retrieval strictness if quality drops.

```python
from phoenix.framework.rag import AdaptiveRAG

async def main():
    adaptive = AdaptiveRAG()
    await adaptive.ingest("./knowledge_base")

    session_id = "user_123"

    # Turn 1
    a1 = await adaptive.query("What is the capital of France?", session_id=session_id)
    print(a1) # "The capital of France is Paris."

    # Turn 2: Contextual reference resolved automatically
    a2 = await adaptive.query("What is the population there?", session_id=session_id)
    print(a2) # Resolves to "What is the population in Paris?" and retrieves context

    # Check session statistics
    stats = adaptive.get_session_stats(session_id)
    print("Session Stats:", stats)
```

---

## 6. MultiModal RAG

Handles mixed media environments. It captions images, transcribes audio, and indexes everything alongside text. It can also answer questions about newly provided images.

```python
from phoenix.framework.rag import MultiModalRAG
from phoenix.services.vlm.openai import OpenAIVLM

async def main():
    # Requires a Vision-Language Model (VLM)
    mmrag = MultiModalRAG(vlm=OpenAIVLM())

    # Ingest a folder containing PDFs, Images (.jpg), and Audio (.mp3)
    # Images are captioned by the VLM, Audio is transcribed by Whisper
    await mmrag.ingest_multimodal("./mixed_media_folder")

    # Standard query across all ingested text, image captions, and audio transcripts
    ans1 = await mmrag.query("Summarize the main points from the audio recording.")

    # Visual QA: Ask a question about an image attachment
    ans2 = await mmrag.query_with_image(
        question="Does this new flowchart match our ingested architecture guidelines?",
        image_path="./new_flowchart.png"
    )
    print(ans2)
```
