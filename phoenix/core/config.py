import os
from dotenv import load_dotenv

# Load .env file if present
load_dotenv(override=True)

class Config:
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    # LLM Config
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    OPENAI_LLM_MODEL = os.getenv("OPENAI_LLM_MODEL", "gpt-4o")
    
    # VLM Config
    OPENAI_VLM_API_KEY = os.getenv("OPENAI_VLM_API_KEY", "")
    OPENAI_VLM_BASE_URL = os.getenv("OPENAI_VLM_BASE_URL", "")
    OPENAI_VLM_MODEL = os.getenv("OPENAI_VLM_MODEL", "")
    QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")

    # Vector DB Config
    VECTOR_DB_TYPE = os.getenv("VECTOR_DB_TYPE", "chroma")
    CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

    # Embedding Config
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    # Local LLM's Config
    # Qwen/Qwen2-1.5B-Instruct
    LOCAL_LLM_TEXT_MODEL = os.getenv("LOCAL_LLM_TEXT_MODEL", "")
    LOCAL_LLM_EMBED_MODEL = os.getenv("LOCAL_LLM_EMBED_MODEL", "all-MiniLM-L6-v2")
    # Qwen/Qwen2-VL-2B-Instruct
    LOCAL_VLM_MODEL = os.getenv("LOCAL_VLM_MODEL", "")
    
    # Deferred Loading Flags
    LOAD_LOCAL_LLM = os.getenv("LOAD_LOCAL_LLM", "false").lower() == "true"
    LOAD_LOCAL_VLM = os.getenv("LOAD_LOCAL_VLM", "false").lower() == "true"
    
    # Fallback Behavior
    AUTO_ACCEPT_FALLBACK = os.getenv("AUTO_ACCEPT_FALLBACK", "false").lower() == "true"
    
    # Training Config
    FINETUNE_PROVIDER = os.getenv("FINETUNE_PROVIDER", "local") # "local" or "openai"
    TRAINING_OUTPUT_DIR = os.getenv("TRAINING_OUTPUT_DIR", "./finetuned_models")
    TRAINING_BATCH_SIZE = int(os.getenv("TRAINING_BATCH_SIZE", "1"))
    TRAINING_EPOCHS = int(os.getenv("TRAINING_EPOCHS", "1"))
    TRAINING_LEARNING_RATE = float(os.getenv("TRAINING_LEARNING_RATE", "2e-4"))
    TRAINING_LORA_R = int(os.getenv("TRAINING_LORA_R", "8"))
    
    # Security & Token Config
    SECURITY_MAX_INPUT_LENGTH = int(os.getenv("SECURITY_MAX_INPUT_LENGTH", "4096"))
    SECURITY_MAX_OUTPUT_LENGTH = int(os.getenv("SECURITY_MAX_OUTPUT_LENGTH", "2048"))
    SECURITY_ENABLE_HALLUCINATION_CHECK = os.getenv("SECURITY_ENABLE_HALLUCINATION_CHECK", "false").lower() == "true"
    
    # RAG Optimization Config
    RAG_QUERY_EXPANSION = os.getenv("RAG_QUERY_EXPANSION", "true").lower() == "true"
    RAG_HYDE_ENABLED = os.getenv("RAG_HYDE_ENABLED", "true").lower() == "true"
    RAG_MMR_ENABLED = os.getenv("RAG_MMR_ENABLED", "true").lower() == "true"
    RAG_MMR_LAMBDA = float(os.getenv("RAG_MMR_LAMBDA", "0.5"))
    RAG_MAX_CONTEXT_CHUNKS = int(os.getenv("RAG_MAX_CONTEXT_CHUNKS", "5"))
    RAG_CONTEXT_COMPRESSION = os.getenv("RAG_CONTEXT_COMPRESSION", "true").lower() == "true"
    
    # New RAG Customization Configs
    RAG_RERANKING_ENABLED = os.getenv("RAG_RERANKING_ENABLED", "true").lower() == "true"
    RAG_FAST_MODE = os.getenv("RAG_FAST_MODE", "false").lower() == "true"
    RAG_DEVICE = os.getenv("RAG_DEVICE", "cpu")
    RAG_SIMILARITY_THRESHOLD = float(os.getenv("RAG_SIMILARITY_THRESHOLD", "0.75"))
    RAG_HYBRID_SEARCH = os.getenv("RAG_HYBRID_SEARCH", "false").lower() == "true"
    RAG_CAG_ENABLED = os.getenv("RAG_CAG_ENABLED", "false").lower() == "true"
    
    # Parent-Child Retrieval Config
    RAG_PARENT_RETRIEVAL = os.getenv("RAG_PARENT_RETRIEVAL", "true").lower() == "true"
    RAG_PARENT_CHUNK_SIZE = int(os.getenv("RAG_PARENT_CHUNK_SIZE", "1500"))
    RAG_CHILD_CHUNK_SIZE = int(os.getenv("RAG_CHILD_CHUNK_SIZE", "300"))
    RAG_CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "50"))
    
config = Config()

