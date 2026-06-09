from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class RAGConfig:
    chunk_size: int = 500
    chunk_overlap: int = 50
    parent_chunk_size: int = 1500
    child_chunk_size: int = 300
    parent_retrieval: bool = True
    top_k: int = 5
    similarity_threshold: float = 0.75
    hyde_enabled: bool = True
    query_expansion: bool = True
    mmr_enabled: bool = True
    mmr_lambda: float = 0.5
    context_compression: bool = True
    reranking: bool = True
    hybrid_search: bool = False
    fast_mode: bool = False
    cache_ttl: int = 300
    max_input_tokens: int = 4096
    max_output_tokens: int = 2048
    system_prompt: Optional[str] = None
    collection_name: str = "phoenix_rag"
    device: str = "cpu"


@dataclass
class CAGConfig(RAGConfig):
    semantic_cache_threshold: float = 0.92
    cache_first: bool = True
    preload_corpus: bool = False
    max_cache_entries: int = 10000


@dataclass
class AgenticRAGConfig(RAGConfig):
    max_retries: int = 3
    confidence_threshold: float = 0.6
    verify_answer: bool = True
    rewrite_on_fail: bool = True
    max_tool_calls: int = 5
    routing_enabled: bool = True


@dataclass
class MultiModalRAGConfig(RAGConfig):
    image_max_size: int = 1024
    image_quality: int = 80
    supported_media: List[str] = field(default_factory=lambda: ["image", "pdf", "audio"])
    ocr_enabled: bool = False
    caption_images: bool = True
