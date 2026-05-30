import asyncio
from typing import Optional, Union, List, Dict, Any
from phoenix.main import (
    init_phoenix, 
    startup_phoenix, 
    get_rag_pipeline, 
    get_insight_engine, 
    get_vlm_pipeline,
    container
)
from phoenix.core.config import config
from phoenix.services.observability.logger import get_logger
from phoenix.core.security import SecurityGuard, SecurityError

logger = get_logger("Phoenix AI.Framework")

class ChatBot:
    """
    High-level Builder for Phoenix AI ChatBots.
    Enables one-liner creation of complex AI agents.
    """
    def __init__(self, local: bool = True, vlm: bool = False, tts: bool = False, stt: bool = False):
        self.local = local
        self.vlm_enabled = vlm
        self.tts_enabled = tts
        self.stt_enabled = stt
        self._rag_path: Optional[str] = None
        self._memory_enabled: bool = False
        self._session_id: str = "default_session"
        self._system_prompt: Optional[str] = None
        self._llm_model: Optional[str] = None
        self._vlm_model: Optional[str] = None
        self._rag_config: Dict[str, Any] = {"chunk_size": 500, "chunk_overlap": 50}
        self._security_enabled: bool = False
        self._security_mode: str = "standard"
        self._openai_key: Optional[str] = None
        self._openai_url: Optional[str] = None

    def with_rag(self, data_to_insight_path: Union[str, List[str]], chunk_size: int = 500, chunk_overlap: int = 50, 
                 reranking: Optional[bool] = None, fast_rag: Optional[bool] = None, device: Optional[str] = None, 
                 threshold: Optional[float] = None, hybrid_search: Optional[bool] = None, cag: Optional[bool] = None):
        """Enable RAG and specify the data path(s) for ingestion."""
        self._rag_path = data_to_insight_path
        self._rag_config = {
            "chunk_size": chunk_size, 
            "chunk_overlap": chunk_overlap,
            "reranking": reranking,
            "fast_rag": fast_rag,
            "device": device,
            "threshold": threshold,
            "hybrid_search": hybrid_search,
            "cag": cag
        }
        return self

    def with_memory(self):
        """Enable conversation history and semantic memory."""
        self._memory_enabled = True
        return self

    def with_system_prompt(self, prompt: str):
        """Set a custom system prompt to guide the bot's behavior."""
        self._system_prompt = prompt
        return self

    def with_model(self, llm: Optional[str] = None, vlm: Optional[str] = None):
        """Override the default LLM or VLM models."""
        self._llm_model = llm
        self._vlm_model = vlm
        return self

    def with_security(self, mode: str = "standard"):
        """Enable the security guard layer (standard or strict)."""
        self._security_enabled = True
        self._security_mode = mode
        return self

    def with_openai(self, api_key: str, base_url: Optional[str] = None):
        """Configure OpenAI credentials and switch to remote mode."""
        self._openai_key = api_key
        self._openai_url = base_url
        self.local = False
        return self

    def set_session(self, session_id: str):
        """Set a specific session ID for memory."""
        self._session_id = session_id
        return self

    def build(self):
        """Initialize the SDK and return a functional ChatBot instance."""
        # 1. Initialize Registry (Sync)
        if not self.local:
            config.LOCAL_LLM_TEXT_MODEL = ""
            config.LOCAL_VLM_MODEL = ""
        
        # Override models if provided
        if self._llm_model:
            if self.local: config.LOCAL_LLM_TEXT_MODEL = self._llm_model
            else: config.OPENAI_LLM_MODEL = self._llm_model
        
        if self._vlm_model:
            if self.local: config.LOCAL_VLM_MODEL = self._vlm_model
            else: config.OPENAI_VLM_MODEL = self._vlm_model
            
        # Apply OpenAI Credentials if provided
        if self._openai_key:
            config.OPENAI_API_KEY = self._openai_key
        if self._openai_url:
            config.OPENAI_BASE_URL = self._openai_url

        init_phoenix()
        
        return ChatBotInstance(self)

class ChatBotInstance:
    """
    The operational ChatBot instance.
    Handles the interaction orchestration.
    """
    def __init__(self, builder: ChatBot):
        self.builder = builder
        self._initialized = False
        self._rag_pipeline = None
        self._vlm_pipeline = None
        self._memory = None
        self._stt = None
        self._tts = None
        self._guard = SecurityGuard(mode=builder._security_mode) if builder._security_enabled else None

    async def _lazy_init(self):
        if self._initialized:
            return

        # 1. Startup Services (Async)
        await startup_phoenix()

        # 2. Setup Components
        if self.builder._rag_path:
            self._rag_pipeline = get_rag_pipeline(rag_config=self.builder._rag_config)
            paths = self.builder._rag_path if isinstance(self.builder._rag_path, list) else [self.builder._rag_path]
            for p in paths:
                logger.info(f"Ingesting: {p}")
                await self._rag_pipeline.ingest(
                    p,
                    chunk_size=self.builder._rag_config["chunk_size"],
                    chunk_overlap=self.builder._rag_config["chunk_overlap"]
                )
        
        if self.builder.vlm_enabled:
            self._vlm_pipeline = get_vlm_pipeline(prefer_local=self.builder.local)

        if self.builder._memory_enabled:
            self._memory = container.get("memory")

        # if self.builder.stt_enabled:
        #     self._stt = container.get("stt_local") if self.builder.local else container.get("stt_openai")
        # 
        # if self.builder.tts_enabled:
        #     self._tts = container.get("tts_local") if self.builder.local else container.get("tts_openai")

        self._initialized = True
        logger.info("ChatBotInstance lazily initialized.")

    def set_session(self, session_id: str):
        """Switch the current session ID for this instance."""
        self.builder._session_id = session_id
        return self

    async def chat(self, 
                   text: Optional[str] = None, 
                   image_path: Optional[str] = None, 
                   audio_path: Optional[str] = None) -> Union[str, Dict[str, Any]]:
        """
        Send a message to the bot.
        Returns the text response, or a dict if multiple modalities are involved.
        """
        await self._lazy_init()

        # 0. Security Inbound Check
        if self._guard:
            try:
                if text:
                    text = await self._guard.validate_input(text)
                if audio_path:
                    # Basic check for audio path validity
                    pass
            except SecurityError as e:
                logger.error(f"Security Block: {e}")
                return f"Security Violation: {str(e)}"

        # 1. Handle Audio Input (STT) - Temporarily Disabled
        # ... (stashed logic)

        if not text and not image_path:
            raise ValueError("Either text, image_path, or audio_path must be provided.")

        # 2. Context Retrieval (Memory)
        context = ""
        if self._memory:
            if hasattr(self._memory, "get_full_context"):
                # Use Advanced Hybrid Memory Structure
                context = await self._memory.get_full_context(self.builder._session_id, query=text)
            else:
                # Fallback to legacy MemoryManager
                history = await self._memory.get_context(self.builder._session_id)
                past_facts = await self._memory.search_memory(self.builder._session_id, text)
                context = f"{past_facts}\n\nRecent History:\n{history}"

        # 3. Decision Logic: VLM vs LLM/RAG
        response_text = ""
        system_instr = self.builder._system_prompt or ""

        if image_path and self._vlm_pipeline:
            # VLM path
            response_text = await self._vlm_pipeline.ask(
                text, 
                image_path, 
                use_rag=bool(self._rag_pipeline),
                session_id=self.builder._session_id,
                system_prompt=system_instr,
                history=context
            )
        elif self._rag_pipeline:
            # RAG path - Pass system_prompt and history separately
            response_text = await self._rag_pipeline.query(text, session_id=self.builder._session_id, system_prompt=system_instr, history=context)
        else:
            # Simple LLM path
            llm = container.get("llm")
            full_prompt = f"{system_instr}\n\nContext: Use the following context if relevant.\n{context}\n\nUser: {text}"
            response_text = await llm.generate(full_prompt)

        # 4. Handle Memory Update
        if self._memory:
            import inspect
            sig = inspect.signature(self._memory.add_interaction)
            if 'role' in sig.parameters:
                # HybridMemoryManager uses (session_id, role, content)
                await self._memory.add_interaction(self.builder._session_id, "user", text)
                await self._memory.add_interaction(self.builder._session_id, "assistant", response_text)
            else:
                # Legacy MemoryManager uses (session_id, prompt, response)
                await self._memory.add_interaction(self.builder._session_id, text, response_text)

        # 4b. Security Outbound Check (Masking)
        if self._guard:
            response_text = self._guard.mask_secrets(response_text)

        # 5. Handle Audio Output (TTS) - Temporarily Disabled
        # ... (stashed logic)
        
        return response_text

