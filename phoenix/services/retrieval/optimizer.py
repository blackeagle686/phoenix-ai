from phoenix.core.config import config
import math

class Optimizer:
    """Handles query manipulation, semantic cache hits, and reranking."""
    
    def rerank(self, docs, query: str):
        """
        Reranks documents using Max Marginal Relevance (MMR) to reduce redundancy.
        """
        if not docs or not config.RAG_MMR_ENABLED:
            return docs[:config.RAG_MAX_CONTEXT_CHUNKS]

        # Standard MMR implementation
        # For simplicity without heavy numpy, we use a basic relevance score (distance) 
        # and a simple penalty for redundancy if we had embeddings here.
        # Since we only have distances from VectorDB, we'll implement a 'diversity' filter.
        
        selected = []
        remaining = list(docs)
        
        # 1. Take the most relevant document first
        selected.append(remaining.pop(0))
        
        while remaining and len(selected) < config.RAG_MAX_CONTEXT_CHUNKS:
            best_mmr = -float('inf')
            best_idx = -1
            
            for i, doc in enumerate(remaining):
                # Relevance: Inverse of distance (lower distance = higher relevance)
                relevance = 1.0 / (doc.get("distance", 0.001) + 1e-6)
                
                # Diversity: Penalty for similarity to already selected docs
                # (Simple token overlap as a proxy for similarity if embeddings aren't accessible)
                max_sim = 0
                for s_doc in selected:
                    sim = self._get_similarity(doc.get("content", ""), s_doc.get("content", ""))
                    if sim > max_sim:
                        max_sim = sim
                
                # MMR formula: lambda * relevance - (1 - lambda) * max_similarity
                mmr_score = config.RAG_MMR_LAMBDA * relevance - (1 - config.RAG_MMR_LAMBDA) * max_sim
                
                if mmr_score > best_mmr:
                    best_mmr = mmr_score
                    best_idx = i
            
            if best_idx != -1:
                selected.append(remaining.pop(best_idx))
            else:
                break
                
        return selected

    def _get_similarity(self, text1: str, text2: str) -> float:
        """Simple Jaccard similarity (token overlap) as a diversity proxy."""
        s1 = set(text1.lower().split())
        s2 = set(text2.lower().split())
        if not s1 or not s2: return 0.0
        return len(s1.intersection(s2)) / len(s1.union(s2))

    async def expand_query(self, query: str, llm=None) -> list:
        """
        Generates 2-3 alternative versions of the query to improve retrieval.
        """
        if not config.RAG_QUERY_EXPANSION or not llm:
            return [query]
            
        prompt = f"""Generate 2 alternative search queries for the following question to help find more relevant information in a knowledge base.
Return ONLY the queries, one per line, no numbering.

Question: {query}"""
        
        try:
            resp = await llm.generate(prompt)
            alternatives = [q.strip() for q in resp.split("\n") if q.strip()]
            return [query] + alternatives[:2]
        except Exception:
            return [query]

    async def get_hyde_query(self, query: str, llm=None) -> str:
        """
        Generates a hypothetical answer (HyDE) to improve embedding matching.
        """
        if not config.RAG_HYDE_ENABLED or not llm:
            return query
            
        prompt = f"""Write a short, hypothetical technical answer to the following question. 
Do not worry about accuracy, just use technical terms and keywords that would likely appear in a document about this topic.

Question: {query}
Hypothetical Answer:"""
        
        try:
            hyde_answer = await llm.generate(prompt)
            return hyde_answer.strip()
        except Exception:
            return query

    def compress_context(self, docs: list, query: str) -> list:
        """
        Strips irrelevant sentences from retrieved chunks to focus the prompt.
        Uses simple keyword matching for speed.
        """
        if not config.RAG_CONTEXT_COMPRESSION or not docs:
            return docs
            
        query_words = set(query.lower().split())
        compressed_docs = []
        
        for doc in docs:
            content = doc.get("content", "")
            # Split into sentences
            sentences = content.replace("\n", ". ").split(". ")
            relevant_sentences = []
            
            for sentence in sentences:
                sentence_words = set(sentence.lower().split())
                # If sentence shares keywords with query, it's likely relevant
                if sentence_words.intersection(query_words):
                    relevant_sentences.append(sentence)
            
            # If we filtered too much, keep some original context
            if len(relevant_sentences) < 2 and len(sentences) > 2:
                # Keep first and last sentence as fallback
                relevant_sentences = [sentences[0], "..."] + relevant_sentences + ["...", sentences[-1]]
            
            doc["content"] = ". ".join(relevant_sentences)
            compressed_docs.append(doc)
            
        return compressed_docs

    def rewrite_query(self, query: str) -> str:
        # Basic query cleaning
        if not query:
            return ""
        query = query.strip()
        return query

