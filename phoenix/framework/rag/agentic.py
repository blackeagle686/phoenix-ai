import json
import hashlib
from typing import Optional, List, Dict, Any
from phoenix.framework.rag.base import BaseRAG
from phoenix.framework.rag.config import AgenticRAGConfig
from phoenix.services.llm.base import BaseLLM
from phoenix.services.vector.base import BaseVectorDB
from phoenix.services.vector.embeddings import BaseEmbeddings
from phoenix.services.observability.logger import get_logger

logger = get_logger("Phoenix AI.AgenticRAG")


class AgenticRAG(BaseRAG):
    """Agentic RAG with autonomous query rewriting, answer verification, 
    routing, and self-correction loops.

    The system acts as an autonomous agent that:
    1. Routes queries to the right retrieval strategy
    2. Rewrites poor queries for better retrieval
    3. Verifies retrieved context relevance before answering
    4. Self-corrects when confidence is low
    5. Supports custom tool functions

    Usage:
        arag = AgenticRAG(
            max_retries=3,
            verify_answer=True,
            rewrite_on_fail=True,
        )
        await arag.ingest("/path/to/docs")
        answer = await arag.query("What is X?")
    """

    def __init__(
        self,
        config: AgenticRAGConfig = None,
        llm: BaseLLM = None,
        vector_db: BaseVectorDB = None,
        embeddings: BaseEmbeddings = None,
        cache=None,
        tools: List[Dict[str, Any]] = None,
        **kwargs
    ):
        cfg = config or AgenticRAGConfig()
        super().__init__(
            config=cfg,
            llm=llm,
            vector_db=vector_db,
            embeddings=embeddings,
            cache=cache,
            **kwargs
        )
        self._tools: Dict[str, callable] = {}
        if tools:
            for t in tools:
                self.register_tool(t["name"], t["fn"], t.get("description", ""))

    def register_tool(self, name: str, fn: callable, description: str = ""):
        self._tools[name] = {"fn": fn, "description": description}

    def _get_tools_schema(self) -> str:
        if not self._tools:
            return ""
        lines = []
        for name, info in self._tools.items():
            lines.append(f"- {name}: {info['description']}")
        return "\n".join(lines)

    async def _route_query(self, question: str) -> str:
        """Decide the retrieval strategy: vector_search, web_search, direct_answer, or tool_call."""
        if not self.config.routing_enabled:
            return "vector_search"

        tools_block = ""
        if self._tools:
            tools_block = f"\n- tool_call (available tools: {', '.join(self._tools.keys())})"

        prompt = (
            f"Given the user question below, decide the best retrieval strategy.\n"
            f"Options:\n"
            f"- vector_search (search the knowledge base)\n"
            f"- direct_answer (you can answer without retrieval)\n"
            f"{tools_block}\n\n"
            f"Question: {question}\n\n"
            f"Respond with ONLY a JSON object: {{\"strategy\": \"...\", \"reason\": \"...\"}}"
        )
        try:
            resp = await self.llm.generate(prompt, max_tokens=100)
            resp = resp.strip()
            if resp.startswith("```"):
                resp = resp.split("```")[1]
                if resp.startswith("json"):
                    resp = resp[4:]
            data = json.loads(resp)
            return data.get("strategy", "vector_search")
        except Exception:
            return "vector_search"

    async def _rewrite_query(self, question: str, context: str = "") -> str:
        prompt = (
            f"The following search query did not return good results.\n"
            f"Original query: {question}\n"
        )
        if context:
            prompt += f"Previous retrieval context: {context[:500]}\n"
        prompt += (
            f"Rewrite the query to be more specific and likely to match relevant documents.\n"
            f"Return ONLY the rewritten query, nothing else."
        )
        try:
            rewritten = await self.llm.generate(prompt, max_tokens=150)
            return rewritten.strip().strip('"').strip("'")
        except Exception:
            return question

    async def _verify_context(self, question: str, docs: List[Dict]) -> Dict[str, Any]:
        """Check if retrieved documents are relevant to the question."""
        if not docs:
            return {"relevant": False, "confidence": 0.0, "reason": "No documents retrieved"}

        context_preview = "\n".join(d.get("content", "")[:300] for d in docs[:3])
        prompt = (
            f"Evaluate if the following retrieved context is relevant to answer the question.\n\n"
            f"Question: {question}\n\n"
            f"Retrieved Context:\n{context_preview}\n\n"
            f"Respond with ONLY a JSON object:\n"
            f'{{"relevant": true/false, "confidence": 0.0-1.0, "reason": "..."}}'
        )
        try:
            resp = await self.llm.generate(prompt, max_tokens=150)
            resp = resp.strip()
            if resp.startswith("```"):
                resp = resp.split("```")[1]
                if resp.startswith("json"):
                    resp = resp[4:]
            return json.loads(resp)
        except Exception:
            return {"relevant": True, "confidence": 0.5, "reason": "Verification parse failed"}

    async def _verify_answer(self, question: str, answer: str, docs: List[Dict]) -> Dict[str, Any]:
        """Verify the generated answer is grounded in the retrieved documents."""
        context_preview = "\n".join(d.get("content", "")[:300] for d in docs[:3])
        prompt = (
            f"Verify if the following answer is factually grounded in the provided context.\n\n"
            f"Question: {question}\n"
            f"Answer: {answer}\n\n"
            f"Context:\n{context_preview}\n\n"
            f"Respond with ONLY a JSON object:\n"
            f'{{"grounded": true/false, "confidence": 0.0-1.0, "issues": "..."}}'
        )
        try:
            resp = await self.llm.generate(prompt, max_tokens=200)
            resp = resp.strip()
            if resp.startswith("```"):
                resp = resp.split("```")[1]
                if resp.startswith("json"):
                    resp = resp[4:]
            return json.loads(resp)
        except Exception:
            return {"grounded": True, "confidence": 0.5, "issues": "Verification parse failed"}

    async def _execute_tool(self, tool_name: str, question: str) -> str:
        if tool_name not in self._tools:
            return f"Tool '{tool_name}' not found."
        fn = self._tools[tool_name]["fn"]
        try:
            import inspect
            if inspect.iscoroutinefunction(fn):
                return str(await fn(question))
            return str(fn(question))
        except Exception as e:
            return f"Tool execution error: {e}"

    async def query(self, question: str, system_prompt: str = None, history: str = None) -> str:
        await self._ensure_init()

        strategy = await self._route_query(question)
        logger.info(f"Agentic routing: {strategy}")

        if strategy == "direct_answer":
            sp = system_prompt or self.config.system_prompt or "You are a helpful AI assistant."
            prompt = f"{sp}\n\nQuestion: {question}"
            if history:
                prompt = f"{sp}\n\nHistory:\n{history}\n\nQuestion: {question}"
            return await self.llm.generate(prompt, max_tokens=self.config.max_output_tokens)

        if strategy == "tool_call":
            tool_name = list(self._tools.keys())[0] if self._tools else None
            if tool_name:
                tool_result = await self._execute_tool(tool_name, question)
                prompt = (
                    f"Using the following tool output, answer the user's question.\n\n"
                    f"Tool: {tool_name}\nTool Output: {tool_result}\n\n"
                    f"Question: {question}"
                )
                return await self.llm.generate(prompt, max_tokens=self.config.max_output_tokens)

        current_query = question
        for attempt in range(self.config.max_retries):
            docs = await self.retrieve(current_query)

            verification = await self._verify_context(current_query, docs)
            confidence = verification.get("confidence", 0.0)
            relevant = verification.get("relevant", False)

            if not relevant and self.config.rewrite_on_fail and attempt < self.config.max_retries - 1:
                context_str = "\n".join(d.get("content", "")[:200] for d in docs[:2])
                current_query = await self._rewrite_query(question, context_str)
                logger.info(f"Query rewritten (attempt {attempt+1}): {current_query[:80]}")
                continue

            prompt = self.composer.build_prompt(
                question, docs,
                system_prompt=system_prompt or self.config.system_prompt,
                history=history
            )
            answer = await self.llm.generate(prompt, max_tokens=self.config.max_output_tokens)

            if self.config.verify_answer and confidence >= self.config.confidence_threshold:
                grounding = await self._verify_answer(question, answer, docs)
                if not grounding.get("grounded", True) and attempt < self.config.max_retries - 1:
                    current_query = await self._rewrite_query(question, grounding.get("issues", ""))
                    logger.info(f"Answer not grounded, retrying with rewritten query.")
                    continue

            if self.semantic_cache:
                await self.semantic_cache.add(self.optimizer.rewrite_query(question), answer)
            return answer

        docs = await self.retrieve(current_query)
        prompt = self.composer.build_prompt(
            question, docs,
            system_prompt=system_prompt or self.config.system_prompt,
            history=history
        )
        return await self.llm.generate(prompt, max_tokens=self.config.max_output_tokens)

    async def query_with_trace(self, question: str, system_prompt: str = None) -> Dict[str, Any]:
        """Returns the answer plus the full agentic trace for debugging."""
        await self._ensure_init()
        trace = {"steps": [], "query_rewrites": [], "verifications": []}

        strategy = await self._route_query(question)
        trace["strategy"] = strategy
        trace["steps"].append(f"Routed to: {strategy}")

        if strategy == "direct_answer":
            answer = await self.llm.generate(question, max_tokens=self.config.max_output_tokens)
            trace["steps"].append("Direct LLM answer (no retrieval)")
            return {"answer": answer, "trace": trace}

        current_query = question
        for attempt in range(self.config.max_retries):
            docs = await self.retrieve(current_query)
            verification = await self._verify_context(current_query, docs)
            trace["verifications"].append(verification)

            if not verification.get("relevant", False) and self.config.rewrite_on_fail and attempt < self.config.max_retries - 1:
                current_query = await self._rewrite_query(question)
                trace["query_rewrites"].append(current_query)
                trace["steps"].append(f"Rewrote query to: {current_query[:80]}")
                continue

            prompt = self.composer.build_prompt(question, docs, system_prompt=system_prompt or self.config.system_prompt)
            answer = await self.llm.generate(prompt, max_tokens=self.config.max_output_tokens)
            trace["steps"].append(f"Generated answer (attempt {attempt+1})")

            if self.config.verify_answer:
                grounding = await self._verify_answer(question, answer, docs)
                trace["verifications"].append(grounding)
                if grounding.get("grounded", True):
                    trace["steps"].append("Answer verified as grounded.")
                    return {"answer": answer, "trace": trace}
                trace["steps"].append("Answer not grounded, retrying...")
                continue

            return {"answer": answer, "trace": trace}

        docs = await self.retrieve(current_query)
        prompt = self.composer.build_prompt(question, docs, system_prompt=system_prompt or self.config.system_prompt)
        answer = await self.llm.generate(prompt, max_tokens=self.config.max_output_tokens)
        trace["steps"].append("Final fallback answer after max retries.")
        return {"answer": answer, "trace": trace}
