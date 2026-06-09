import hashlib
from typing import Optional, List, Dict, Any
from phoenix.framework.rag.base import BaseRAG
from phoenix.framework.rag.config import CAGConfig
from phoenix.services.llm.base import BaseLLM
from phoenix.services.vector.base import BaseVectorDB
from phoenix.services.vector.embeddings import BaseEmbeddings
from phoenix.services.cache.semantic import SemanticCache
from phoenix.services.observability.logger import get_logger

logger = get_logger("Phoenix AI.CAG")


class CAG(BaseRAG):
    """Cache-Augmented Generation.
    
    Prioritizes cached answers over retrieval. Designed for high-throughput 
    environments where the same or semantically similar questions are asked 
    repeatedly. Reduces LLM calls and latency dramatically.

    Usage:
        cag = CAG(
            semantic_cache_threshold=0.90,
            cache_first=True,
        )
        await cag.ingest("/path/to/docs")
        answer = await cag.query("What is X?")
    """

    def __init__(
        self,
        config: CAGConfig = None,
        llm: BaseLLM = None,
        vector_db: BaseVectorDB = None,
        embeddings: BaseEmbeddings = None,
        cache=None,
        **kwargs
    ):
        cfg = config or CAGConfig()
        super().__init__(
            config=cfg,
            llm=llm,
            vector_db=vector_db,
            embeddings=embeddings,
            cache=cache,
            **kwargs
        )
        self.semantic_cache = SemanticCache(
            embeddings=self.embeddings,
            threshold=self.config.semantic_cache_threshold
        )
        self._corpus_cache: Dict[str, str] = {}

    async def preload_corpus(self, questions: List[str]):
        """Pre-generate and cache answers for known frequent questions."""
        await self._ensure_init()
        for q in questions:
            hit = await self.semantic_cache.get_similar(q)
            if hit:
                continue
            answer = await self._generate_answer(q)
            await self.semantic_cache.add(q, answer)
            self._corpus_cache[q] = answer
        logger.info(f"Preloaded {len(questions)} Q&A pairs into semantic cache.")

    async def _generate_answer(self, question: str) -> str:
        docs = await self.retrieve(question)
        prompt = self.composer.build_prompt(question, docs, system_prompt=self.config.system_prompt)
        return await self.llm.generate(prompt, max_tokens=self.config.max_output_tokens)

    async def query(self, question: str, system_prompt: str = None, history: str = None) -> str:
        await self._ensure_init()
        optimized = self.optimizer.rewrite_query(question)

        hit = await self.semantic_cache.get_similar(optimized)
        if hit:
            logger.info("CAG semantic cache hit. Skipping retrieval and LLM call.")
            return hit

        if self.cache:
            key = f"cag:{hashlib.md5(optimized.encode()).hexdigest()}"
            cached = await self.cache.get(key)
            if cached:
                logger.info("CAG redis cache hit.")
                return cached

        answer = await self._generate_answer(question)

        await self.semantic_cache.add(optimized, answer)
        if self.cache:
            await self.cache.set(key, answer, ttl=self.config.cache_ttl)

        return answer

    async def invalidate(self, question: str = None):
        """Clear specific or all cached entries."""
        if question is None:
            self.semantic_cache._items.clear()
            self._corpus_cache.clear()
            logger.info("All CAG caches invalidated.")
        else:
            self.semantic_cache._items = [
                item for item in self.semantic_cache._items
                if item[0] != question
            ]
            self._corpus_cache.pop(question, None)

    def get_cache_stats(self) -> Dict[str, int]:
        return {
            "semantic_cache_entries": len(self.semantic_cache._items),
            "corpus_cache_entries": len(self._corpus_cache)
        }
