from phoenix.services.retrieval.base import BaseInsightService
from phoenix.services.retrieval.retriever import VectorRetriever
from phoenix.services.retrieval.composer import PromptComposer
from phoenix.services.retrieval.optimizer import Optimizer
from typing import Optional
from phoenix.core.config import config
from phoenix.core.utils import async_confirm
from phoenix.services.observability.logger import get_logger
from phoenix.services.cache.semantic import SemanticCache

logger = get_logger("Phoenix AI.Insight")

class InsightEngine(BaseInsightService):
    """
    Main orchestration layer.
    Manages vector retrieval, prompt building, LLM generation, and cache checking.
    """
    def __init__(self, vector_db, primary, fallback, cache=None, semantic_cache: Optional[SemanticCache] = None, rag_config: dict = None):
        self.vector_db = vector_db
        self.primary = primary
        self.fallback = fallback
        self.cache = cache
        self.semantic_cache = semantic_cache
        self.rag_config = rag_config or {}
        
        self.retriever = VectorRetriever(vector_db)
        self.composer = PromptComposer()
        self.optimizer = Optimizer()

    async def init(self):
        pass

    async def query(self, question: str, context: Optional[dict] = None, system_prompt: str = None, history: str = None):
        optimized_query = self.optimizer.rewrite_query(question)
        if optimized_query != question:
            logger.info(f"Query optimized: {question} -> {optimized_query}")
        
        # 0. Selection: Choose provider (Primary preferred)
        provider = self.primary
        if hasattr(provider, "is_available") and not provider.is_available():
            confirmed = await async_confirm("Primary LLM provider is unavailable. Switch to Fallback?")
            if not confirmed:
                return "Operation cancelled by user: Primary unavailable and Fallback rejected."
            provider = self.fallback

        # Optimization Flags
        fast_rag = self.rag_config.get("fast_rag")
        if fast_rag is None:
            fast_rag = config.RAG_FAST_MODE
            
        reranking = self.rag_config.get("reranking")
        if reranking is None:
            reranking = config.RAG_RERANKING_ENABLED
            
        cag = self.rag_config.get("cag")
        if cag is None:
            cag = config.RAG_CAG_ENABLED
            
        hybrid_search = self.rag_config.get("hybrid_search")
        if hybrid_search is None:
            hybrid_search = config.RAG_HYBRID_SEARCH

        # 1. Cache check (Fast path / CAG)
        if self.semantic_cache:
            if cag:
                logger.info("CAG is enabled. Relying strongly on Cache-Augmented Generation.")
            semantic_hit = await self.semantic_cache.get_similar(optimized_query)
            if semantic_hit:
                logger.info("Semantic cache hit (similarity threshold met).")
                return semantic_hit

        cache_key = f"insight:{optimized_query}"
        if self.cache:
            cached = await self.cache.get(cache_key)
            if cached:
                logger.info(f"Insight Cache Hit for key: {cache_key}")
                return cached
            logger.info(f"Insight Cache Miss for key: {cache_key}")

        # 2. Vector retrieval & Query Expansion
        logger.info(f"Retrieving Knowledge for: {optimized_query[:50]}... (fast_rag={fast_rag}, hybrid={hybrid_search})")
        
        search_query = optimized_query
        if not fast_rag and config.RAG_HYDE_ENABLED:
            search_query = await self.optimizer.get_hyde_query(optimized_query, llm=provider)
            logger.info(f"HyDE Answer Generated: {search_query[:50]}...")

        # Expand query (Multi-query search)
        if not fast_rag and config.RAG_QUERY_EXPANSION:
            queries = await self.optimizer.expand_query(search_query, llm=provider)
        else:
            queries = [search_query]
        
        all_docs = []
        seen_contents = set()
        
        for q in queries:
            # Pass hybrid flag to retriever if supported
            if hasattr(self.retriever, "retrieve"):
                docs = await self.retriever.retrieve(q, hybrid=hybrid_search)
            else:
                docs = []
            for d in docs:
                content = d.get("content", "")
                if content not in seen_contents:
                    all_docs.append(d)
                    seen_contents.add(content)
        
        if all_docs:
            if reranking:
                logger.info(f"Retrieved {len(all_docs)} unique documents. Reranking...")
                # MMR Reranking
                all_docs = self.optimizer.rerank(all_docs, optimized_query)
            else:
                logger.info(f"Retrieved {len(all_docs)} unique documents. Reranking skipped.")
                
            if not fast_rag and config.RAG_CONTEXT_COMPRESSION:
                # Contextual Compression
                docs = self.optimizer.compress_context(all_docs, optimized_query)
            else:
                docs = all_docs
        else:
            docs = []

        # 3. Prompt construction
        prompt = self.composer.build_prompt(question, docs, system_prompt=system_prompt, history=history)
        
        # Enforce input length limit
        if len(prompt.split()) > config.SECURITY_MAX_INPUT_LENGTH * 1.5: # Rough estimate
            # Very basic truncation for now
            prompt = prompt[:config.SECURITY_MAX_INPUT_LENGTH * 4]

        # 4. LLM Generation (with runtime fallback)
        try:
            response = await provider.generate(prompt)
        except Exception as e:
            logger.error(f"LLM provider failed: {e}")
            if provider == self.primary and self.fallback:
                confirmed = config.AUTO_ACCEPT_FALLBACK
                if not confirmed:
                    confirmed = await async_confirm(f"Primary LLM failed ({e}). Switch to Fallback for this request?")
                
                if confirmed:
                    if hasattr(self.fallback, "is_available") and not self.fallback.is_available():
                        return f"Error: Secondary provider is not configured. Cannot fallback."
                    
                    logger.info("Retrying with secondary provider...")
                    try:
                        response = await self.fallback.generate(prompt)
                    except Exception as fallback_e:
                        return f"Error: Both primary and fallback providers failed.\nPrimary: {e}\nFallback: {fallback_e}"
                else:
                    return f"Error: Primary LLM failed and fallback was rejected. Details: {e}"
            else:
                raise e

        # 5. Response caching
        if self.semantic_cache:
            await self.semantic_cache.add(optimized_query, response)
        if self.cache:
            await self.cache.set(cache_key, response, ttl=300)

        return response

