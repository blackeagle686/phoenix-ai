from typing import List, Optional, Dict, Any
from phoenix.framework.agent.memory.short_term.stm_manager import ShortTermMemoryManager as ConversationHistory
from phoenix.framework.chatbot.memory.semantic.semantic_search import SemanticSearch as SemanticMemory

class MemoryManager:
    """
    Unified manager for short-term and long-term memory.
    Updated to use new modular components.
    """
    def __init__(self, vector_db=None):
        self.history = ConversationHistory()
        self.semantic = SemanticMemory(vector_db)

    async def add_interaction(self, session_id: str, prompt: str, response: str, metadata: Optional[Dict] = None):
        """Adds a full interaction to both history and semantic memory."""
        await self.history.add(session_id, {"role": "user", "content": prompt}, metadata)
        await self.history.add(session_id, {"role": "assistant", "content": response}, metadata)
        await self.semantic.add(session_id, f"User: {prompt}\nAssistant: {response}", metadata)

    async def get_context(self, session_id: str, limit: int = 5) -> str:
        """Retrieves consolidated context for a given session."""
        return await self.history.get_context_string(session_id)

    async def search_memory(self, session_id: str, query: str, limit: int = 3) -> str:
        """Searches semantic memory for relevant past facts."""
        results = await self.semantic.search(session_id, query, limit=limit)
        if not results:
            return ""
            
        context = "\nRelevant past information:\n"
        for res in results:
            content = res.get("content", res.get("text", str(res))) if isinstance(res, dict) else str(res)
            context += f"- {content}\n"
        return context

    async def clear_session(self, session_id: str):
        """Clears both short-term and semantic memory for a session."""
        await self.history.clear(session_id)
        await self.semantic.clear(session_id)

