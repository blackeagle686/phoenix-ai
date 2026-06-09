from typing import Optional, List, Dict, Any
from phoenix.framework.rag.base import BaseRAG
from phoenix.framework.rag.config import RAGConfig
from phoenix.services.llm.base import BaseLLM
from phoenix.services.vector.base import BaseVectorDB
from phoenix.services.vector.embeddings import BaseEmbeddings
from phoenix.services.observability.logger import get_logger

logger = get_logger("Phoenix AI.RAG")


class RAG(BaseRAG):
    """Standard and Advanced RAG system.
    
    Usage:
        rag = RAG(
            chunk_size=500,
            hyde_enabled=True,
            reranking=True,
            query_expansion=True,
            mmr_enabled=True,
            context_compression=True,
        )
        await rag.ingest("/path/to/docs")
        answer = await rag.query("What is X?")
    """

    def __init__(
        self,
        config: RAGConfig = None,
        llm: BaseLLM = None,
        vector_db: BaseVectorDB = None,
        embeddings: BaseEmbeddings = None,
        cache=None,
        **kwargs
    ):
        super().__init__(
            config=config,
            llm=llm,
            vector_db=vector_db,
            embeddings=embeddings,
            cache=cache,
            **kwargs
        )

    async def query(self, question: str, system_prompt: str = None, history: str = None) -> str:
        return await super().query(question, system_prompt=system_prompt, history=history)

    async def query_with_sources(self, question: str, system_prompt: str = None, history: str = None) -> Dict[str, Any]:
        await self._ensure_init()
        docs = await self.retrieve(question)
        prompt = self.composer.build_prompt(
            question, docs,
            system_prompt=system_prompt or self.config.system_prompt,
            history=history
        )
        response = await self.llm.generate(prompt, max_tokens=self.config.max_output_tokens)
        sources = []
        for d in docs:
            meta = d.get("metadata", {})
            sources.append({
                "source": meta.get("source", "unknown"),
                "path": meta.get("path", ""),
                "content_preview": d.get("content", "")[:200]
            })
        return {"answer": response, "sources": sources}

    async def query_stream(self, question: str, system_prompt: str = None, history: str = None):
        await self._ensure_init()
        docs = await self.retrieve(question)
        prompt = self.composer.build_prompt(
            question, docs,
            system_prompt=system_prompt or self.config.system_prompt,
            history=history
        )
        if hasattr(self.llm, "generate_stream"):
            async for chunk in self.llm.generate_stream(prompt, max_tokens=self.config.max_output_tokens):
                yield chunk
        else:
            result = await self.llm.generate(prompt, max_tokens=self.config.max_output_tokens)
            yield result
