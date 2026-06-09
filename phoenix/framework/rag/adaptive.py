import json
import hashlib
from typing import Optional, List, Dict, Any
from phoenix.framework.rag.base import BaseRAG
from phoenix.framework.rag.config import RAGConfig
from phoenix.services.llm.base import BaseLLM
from phoenix.services.vector.base import BaseVectorDB
from phoenix.services.vector.embeddings import BaseEmbeddings
from phoenix.services.cache.semantic import SemanticCache
from phoenix.services.observability.logger import get_logger

logger = get_logger("Phoenix AI.AdaptiveRAG")


class AdaptiveRAG(BaseRAG):
    """Adaptive RAG with persistent memory, self-correction, 
    and dynamic retrieval strategy adjustment.

    Features:
    - Multi-turn conversation memory
    - Self-correction when retrieval quality drops
    - Dynamic switching between fast/deep retrieval
    - Confidence-based fallback strategies
    - Conversation-aware context building

    Usage:
        arag = AdaptiveRAG()
        await arag.ingest("/path/to/docs")
        a1 = await arag.query("What is X?", session_id="user1")
        a2 = await arag.query("Tell me more about that", session_id="user1")
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
        self._sessions: Dict[str, List[Dict[str, str]]] = {}
        self._quality_tracker: Dict[str, List[float]] = {}

    def _get_history(self, session_id: str) -> List[Dict[str, str]]:
        return self._sessions.get(session_id, [])

    def _add_turn(self, session_id: str, role: str, content: str):
        if session_id not in self._sessions:
            self._sessions[session_id] = []
        self._sessions[session_id].append({"role": role, "content": content})
        if len(self._sessions[session_id]) > 20:
            self._sessions[session_id] = self._sessions[session_id][-20:]

    def _format_history(self, session_id: str) -> str:
        turns = self._get_history(session_id)
        if not turns:
            return ""
        return "\n".join(f"{t['role'].title()}: {t['content']}" for t in turns[-8:])

    async def _assess_quality(self, question: str, docs: List[Dict]) -> float:
        if not docs:
            return 0.0
        context = "\n".join(d.get("content", "")[:300] for d in docs[:3])
        prompt = (
            f"Rate the relevance of the following context to the question on a scale of 0.0 to 1.0.\n\n"
            f"Question: {question}\n\nContext:\n{context}\n\n"
            f"Respond with ONLY a JSON object: {{\"score\": 0.0}}"
        )
        try:
            resp = await self.llm.generate(prompt, max_tokens=50)
            resp = resp.strip()
            if resp.startswith("```"):
                resp = resp.split("```")[1]
                if resp.startswith("json"):
                    resp = resp[4:]
            return float(json.loads(resp).get("score", 0.5))
        except Exception:
            return 0.5

    def _track_quality(self, session_id: str, score: float):
        if session_id not in self._quality_tracker:
            self._quality_tracker[session_id] = []
        self._quality_tracker[session_id].append(score)
        if len(self._quality_tracker[session_id]) > 10:
            self._quality_tracker[session_id] = self._quality_tracker[session_id][-10:]

    def _should_use_deep_retrieval(self, session_id: str) -> bool:
        scores = self._quality_tracker.get(session_id, [])
        if len(scores) < 2:
            return False
        avg = sum(scores[-3:]) / len(scores[-3:])
        return avg < 0.5

    async def _resolve_reference(self, question: str, session_id: str) -> str:
        """Resolve pronouns and references using conversation history."""
        history = self._format_history(session_id)
        if not history:
            return question

        prompt = (
            f"Given the conversation history, rewrite the following question to be self-contained "
            f"(resolve pronouns like 'it', 'that', 'they' and implicit references).\n\n"
            f"History:\n{history}\n\n"
            f"Current question: {question}\n\n"
            f"Return ONLY the rewritten question."
        )
        try:
            rewritten = await self.llm.generate(prompt, max_tokens=200)
            resolved = rewritten.strip().strip('"').strip("'")
            if len(resolved) > 10:
                return resolved
        except Exception:
            pass
        return question

    async def query(self, question: str, session_id: str = None, system_prompt: str = None) -> str:
        await self._ensure_init()
        sid = session_id or "default"

        self._add_turn(sid, "user", question)
        resolved = await self._resolve_reference(question, sid)
        history_str = self._format_history(sid)

        use_deep = self._should_use_deep_retrieval(sid)
        if use_deep:
            logger.info("Adaptive: switching to deep retrieval due to low quality scores.")
            original_fast = self.config.fast_mode
            original_hyde = self.config.hyde_enabled
            original_expand = self.config.query_expansion
            self.config.fast_mode = False
            self.config.hyde_enabled = True
            self.config.query_expansion = True

        optimized = self.optimizer.rewrite_query(resolved)

        if self.semantic_cache:
            hit = await self.semantic_cache.get_similar(optimized)
            if hit:
                self._add_turn(sid, "assistant", hit)
                return hit

        search_query = optimized
        if not self.config.fast_mode and self.config.hyde_enabled:
            search_query = await self.optimizer.get_hyde_query(optimized, llm=self.llm)

        docs = await self.retrieve(search_query)
        quality = await self._assess_quality(resolved, docs)
        self._track_quality(sid, quality)

        if quality < 0.3 and not use_deep:
            logger.info(f"Low quality ({quality:.2f}), retrying with expanded search.")
            expanded = await self.optimizer.expand_query(resolved, llm=self.llm)
            all_docs = list(docs)
            seen = set(d.get("content", "") for d in all_docs)
            for eq in expanded:
                extra = await self.retriever.retrieve(eq, hybrid=self.config.hybrid_search)
                for d in extra:
                    c = d.get("content", "")
                    if c not in seen:
                        all_docs.append(d)
                        seen.add(c)
            docs = all_docs

        prompt = self.composer.build_prompt(
            question, docs,
            system_prompt=system_prompt or self.config.system_prompt,
            history=history_str
        )
        answer = await self.llm.generate(prompt, max_tokens=self.config.max_output_tokens)

        if use_deep:
            self.config.fast_mode = original_fast
            self.config.hyde_enabled = original_hyde
            self.config.query_expansion = original_expand

        self._add_turn(sid, "assistant", answer)

        if self.semantic_cache:
            await self.semantic_cache.add(optimized, answer)

        return answer

    def clear_session(self, session_id: str):
        self._sessions.pop(session_id, None)
        self._quality_tracker.pop(session_id, None)

    def clear_all_sessions(self):
        self._sessions.clear()
        self._quality_tracker.clear()

    def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        return {
            "turns": len(self._sessions.get(session_id, [])),
            "quality_scores": self._quality_tracker.get(session_id, []),
            "avg_quality": (
                sum(self._quality_tracker.get(session_id, [])) / len(self._quality_tracker[session_id])
                if self._quality_tracker.get(session_id) else 0.0
            )
        }
