from typing import Any, Callable, Optional
from phoenix.core.container import container
from phoenix.core.config import config
from phoenix.services.cache.redis_cache import RedisCache
from phoenix.services.llm.openai import OpenAILLM
from phoenix.services.llm.local import LocalLLM
from phoenix.services.vector.chroma import ChromaVectorDB
from phoenix.services.vector.qdrant import QdrantVectorDB
from phoenix.services.vector.embeddings import SentenceTransformerEmbeddings
from phoenix.services.vlm.openai import OpenAIVLM
from phoenix.services.vlm.local import LocalVLM
from phoenix.services.retrieval.vlm_pipeline import VLMPipeline
from phoenix.services.retrieval.engine import InsightEngine
from phoenix.services.rag.pipeline import RAGPipeline
from phoenix.services.training.local_finetuner import LocalFineTuner
from phoenix.services.training.openai_finetuner import OpenAIFineTuner
# from phoenix.framework.chatbot.memory.manager import MemoryManager  # Moved to functions to break cycle
from phoenix.services.audio.local import LocalSTT, LocalTTS
from phoenix.services.audio.openai import OpenAISTT, OpenAITTS
from phoenix.services.observability.logger import get_logger
from phoenix.services.cache.semantic import SemanticCache
import asyncio

logger = get_logger("Phoenix AI.Main")

def init_phoenix(local: bool = False, vlm: bool = False):
    # 0. Set local loading preferences
    config.LOAD_LOCAL_LLM = local
    config.LOAD_LOCAL_VLM = vlm

    # Hardware check for local models
    if local or vlm:
        try:
            from phoenix.core.hardware_check import HardwareChecker
            HardwareChecker.check_all()
        except ImportError:
            # Fallback if psutil is not yet installed
            pass


    # 1. Register Cache
    container.register("cache", RedisCache())
    
    # 2. Register LLM Providers
    llm_openai = OpenAILLM()
    llm_local = LocalLLM()
    container.register("llm_openai", llm_openai)
    container.register("llm_local", llm_local)
    
    # 2b. Generic 'llm' selection
    if local:
        container.register("llm", llm_local)
    else:
        # If not local, we MUST use OpenAI (or its mock/error state) 
        # to avoid crashing on unloaded local models
        container.register("llm", llm_openai)

    # 3. Register Embeddings
    embeddings = SentenceTransformerEmbeddings()
    container.register("embeddings", embeddings)

    # 4. Register VLM Providers
    vlm_openai = OpenAIVLM()
    vlm_local = LocalVLM()
    container.register("vlm_openai", vlm_openai)
    container.register("vlm_local", vlm_local)
    
    # 4b. Generic 'vlm' selection
    if vlm:
        container.register("vlm", vlm_local)
    else:
        container.register("vlm", vlm_openai)
    
    # 5. Register Vector DB based on config
    if config.VECTOR_DB_TYPE == "chroma":
        vector_db = ChromaVectorDB(embedding_service=embeddings)
    elif config.VECTOR_DB_TYPE == "qdrant":
        vector_db = QdrantVectorDB()
    else:
        raise ValueError(f"Unsupported vector DB type: {config.VECTOR_DB_TYPE}")
    
    container.register("vector_db", vector_db)

    # 6. Register Fine-Tuning Services
    container.register("finetune_local", LocalFineTuner())
    container.register("finetune_openai", OpenAIFineTuner())

    # 7. Register Memory Manager
    from phoenix.framework.chatbot.memory.manager import MemoryManager
    container.register("memory", MemoryManager(container.get("vector_db")))

    # 8. Register Audio Services
    container.register("stt_local", LocalSTT())
    container.register("stt_openai", OpenAISTT())
    container.register("tts_local", LocalTTS())
    container.register("tts_openai", OpenAITTS())

async def startup_phoenix(on_progress: Optional[Callable[[float, str], Any]] = None):
    """
    Asynchronously initializes all services registered in the container.
    This includes Cache connections, Vector DB clients, and LLM pools.
    """
    init_tasks = []

    # 1. Start Cache
    cache = container.get("cache")
    if hasattr(cache, "init"):
        init_tasks.append(cache.init())
    
    # 2. Start LLM Providers
    llm_openai = container.get("llm_openai")
    llm_local = container.get("llm_local")
    if hasattr(llm_openai, "init"):
        init_tasks.append(llm_openai.init())
    
    if config.LOAD_LOCAL_LLM and hasattr(llm_local, "init"):
        init_tasks.append(llm_local.init())
    elif not config.LOAD_LOCAL_LLM:
        logger.info("Local LLM loading skipped (LOAD_LOCAL_LLM=False).")
        
    # 3. Start Vector DB
    vector_db = container.get("vector_db")
    if hasattr(vector_db, "init"):
        init_tasks.append(vector_db.init())
    
    # 4. Start VLM Providers
    vlm_openai = container.get("vlm_openai")
    vlm_local = container.get("vlm_local")
    if hasattr(vlm_openai, "init"):
        init_tasks.append(vlm_openai.init())
    
    if config.LOAD_LOCAL_VLM and hasattr(vlm_local, "init"):
        init_tasks.append(vlm_local.init())
    elif not config.LOAD_LOCAL_VLM:
        logger.info("Local VLM loading skipped (LOAD_LOCAL_VLM=False).")
    
    # 5. Start Audio Services
    for service_name in ["stt_local", "stt_openai", "tts_local", "tts_openai"]:
        service = container.get(service_name)
        if hasattr(service, "init"):
            init_tasks.append(service.init())

    if init_tasks:
        if on_progress:
            total = len(init_tasks)
            completed = 0
            
            async def wrapped_task(task):
                nonlocal completed
                await task
                completed += 1
                on_progress(completed / total, f"Started {completed}/{total} services...")
            
            await asyncio.gather(*(wrapped_task(t) for t in init_tasks))
        else:
            await asyncio.gather(*init_tasks)
            
    logger.info("Phoenix AI SDK Services started successfully.")

def get_rag_pipeline(rag_config: dict = None) -> RAGPipeline:
    if not container._registry:
        init_phoenix()
    vector_db = container.get("vector_db")
    primary = container.get("llm")
    llm_openai = container.get("llm_openai")
    llm_local = container.get("llm_local")
    cache = container.get("cache")
    embeddings = container.get("embeddings")
    
    # Configure semantic cache threshold
    threshold = config.RAG_SIMILARITY_THRESHOLD
    if rag_config and rag_config.get("threshold") is not None:
        threshold = rag_config["threshold"]
        
    semantic_cache = SemanticCache(embeddings=embeddings, threshold=threshold)
    
    # Fallback should be the one NOT chosen as primary
    fallback = llm_openai if primary == llm_local else llm_local
    
    return RAGPipeline(vector_db, primary=primary, fallback=fallback, cache=cache, semantic_cache=semantic_cache, rag_config=rag_config)

def get_insight_engine(openai_model: str = None, local_model: str = None, prefer_local: bool = None) -> InsightEngine:
    if not container._registry:
        init_phoenix()
    vector_db = container.get("vector_db")
    llm_openai = container.get("llm_openai")
    llm_local = container.get("llm_local")
    cache = container.get("cache")
    embeddings = container.get("embeddings")
    semantic_cache = SemanticCache(embeddings=embeddings, threshold=0.95)
    
    if openai_model:
        llm_openai.model = openai_model
    if local_model:
        llm_local.model = local_model
        
    if prefer_local is True:
        return InsightEngine(vector_db, llm_local, llm_openai, cache, semantic_cache=semantic_cache)
    elif prefer_local is False:
        return InsightEngine(vector_db, llm_openai, llm_local, cache, semantic_cache=semantic_cache)
        
    # Default: use generic 'llm' as primary
    primary = container.get("llm")
    fallback = llm_openai if primary == llm_local else llm_local
    return InsightEngine(vector_db, primary, fallback, cache, semantic_cache=semantic_cache)

def get_vlm_pipeline(openai_model: str = None, local_model: str = None, prefer_local: bool = True) -> VLMPipeline:
    vlm_openai = container.get("vlm_openai")
    vlm_local = container.get("vlm_local")
    vector_db = container.get("vector_db")
    cache = container.get("cache")
    
    # Dynamic overrides
    if openai_model:
        vlm_openai.model = openai_model
    if local_model:
        vlm_local.model = local_model
    
    if prefer_local:
        # Local is primary, OpenAI is fallback
        return VLMPipeline(vlm_local, vlm_openai, vector_db, cache)
    
    return VLMPipeline(vlm_openai, vlm_local, vector_db, cache)

def get_finetuner(provider: str = None) -> Any:
    """
    Returns the fine-tuning service based on provider ('local' or 'openai').
    Defaults to config.FINETUNE_PROVIDER.
    """
    provider = provider or config.FINETUNE_PROVIDER
    if provider == "openai":
        return container.get("finetune_openai")
    return container.get("finetune_local")

def get_llm() -> Any:
    return container.get("llm")

def get_memory():
    from phoenix.framework.chatbot.memory.manager import MemoryManager
    return container.get("memory")

async def init_phoenix_full(local: bool = False, vlm: bool = False, parallel_hooks: bool = False, on_progress: Optional[Callable] = None):
    """
    Complete initialization:
    1. init_phoenix (Registry)
    2. startup_phoenix (Connections)
    3. lifecycle.startup (Hooks)
    """
    from phoenix.core.lifecycle import lifecycle
    init_phoenix(local=local, vlm=vlm)
    await startup_phoenix(on_progress=on_progress)
    await lifecycle.startup(parallel=parallel_hooks)
    logger.info(f"Phoenix AI SDK initialized (parallel_hooks={parallel_hooks}).")



def set_providers(llm_provider: str = None, vlm_provider: str = None) -> None:
    """
    Explicitly configure which provider to use for generic `llm` and `vlm`.

    llm_provider, vlm_provider: one of 'openai', 'local', or 'auto' (default 'auto').
    - 'openai' forces the OpenAI provider (may raise/print warnings if not available).
    - 'local' forces the Local provider.
    - 'auto' (or None) prefers OpenAI when available, otherwise local.

    Call this after `init_phoenix()` and before `startup_phoenix()` to customize behaviour.
    """
    # Helper to resolve a provider choice
    def resolve(choice, name_openai, name_local):
        if choice == "openai":
            return container.get(name_openai)
        if choice == "local":
            return container.get(name_local)
        # auto or None
        try:
            openai = container.get(name_openai)
            local = container.get(name_local)
            if hasattr(openai, "is_available") and openai.is_available():
                return openai
            return local
        except Exception:
            return container.get(name_local)

    if llm_provider is not None:
        llm = resolve(llm_provider, "llm_openai", "llm_local")
        container.register("llm", llm)

    if vlm_provider is not None:
        vlm = resolve(vlm_provider, "vlm_openai", "vlm_local")
        container.register("vlm", vlm)


def get_providers() -> dict:
    """Return currently configured generic providers for `llm` and `vlm`."""
    res = {}
    try:
        res["llm"] = container.get("llm")
    except Exception:
        res["llm"] = None
    try:
        res["vlm"] = container.get("vlm")
    except Exception:
        res["vlm"] = None
    return res

