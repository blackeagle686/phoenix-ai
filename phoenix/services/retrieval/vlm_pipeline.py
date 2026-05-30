import hashlib
import os
from typing import Optional
from phoenix.services.retrieval.retriever import VectorRetriever
from phoenix.services.retrieval.composer import PromptComposer
from phoenix.services.observability.logger import get_logger
from phoenix.core.utils import async_confirm

logger = get_logger("Phoenix AI.VLM")

class VLMPipeline:
    """
    High-level orchestration for Vision-Language Models.
    Integrates VLM, RAG context, and Caching in a unified interface.
    """
    def __init__(self, primary, fallback, vector_db=None, cache=None):
        self.primary = primary
        self.fallback = fallback
        self.vector_db = vector_db
        self.cache = cache
        
        self.retriever = VectorRetriever(vector_db) if vector_db else None
        self.composer = PromptComposer()

    def _get_cache_key(self, prompt: str, image_path: str) -> str:
        # Create a unique key based on prompt and image content (simplified to path + stats for speed)
        # or just path + hash of prompt
        try:
            mtime = os.path.getmtime(image_path)
            size = os.path.getsize(image_path)
            combined = f"{prompt}:{image_path}:{mtime}:{size}"
            return f"vlm_cache:{hashlib.md5(combined.encode()).hexdigest()}"
        except Exception:
            # Fallback if file access fails
            return f"vlm_cache:{hashlib.md5((prompt + image_path).encode()).hexdigest()}"

    def _preprocess_image(self, image_path: str, max_size: int = 1024, quality: int = 80) -> str:
        """
        Resizes and optimizes an image for VLM processing to save bandwidth and fit model constraints.
        """
        try:
            from PIL import Image
            import tempfile
            
            img = Image.open(image_path)
            orig_w, orig_h = img.size
            
            # 1. Resize if too large
            if orig_w > max_size or orig_h > max_size:
                if orig_w > orig_h:
                    new_w = max_size
                    new_h = int(max_size * orig_h / orig_w)
                else:
                    new_h = max_size
                    new_w = int(max_size * orig_w / orig_h)
                
                img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                logger.info(f"Image resized for VLM: {orig_w}x{orig_h} -> {new_w}x{new_h}")
            
            # 2. Save as optimized JPEG to a temporary file
            temp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            img.convert("RGB").save(temp_file.name, "JPEG", quality=quality, optimize=True)
            return temp_file.name
            
        except Exception as e:
            logger.warning(f"Image preprocessing skipped due to error: {e}")
            return image_path

    async def ask(self, prompt: str, image_path: str, use_rag: bool = False, session_id: Optional[str] = None, system_prompt: str = None, history: str = None) -> str:
        """
        Ask a question about an image, optionally using RAG for context.
        """
        # 1. Selection: Choose provider (Primary preferred if available)
        provider = self.primary
        if hasattr(provider, "is_available") and not provider.is_available():
            confirmed = await async_confirm("Primary VLM provider is unavailable. Switch to Fallback?")
            if not confirmed:
                return "Operation cancelled by user: Primary unavailable and Fallback rejected."
            
            logger.warning("Falling back to secondary VLM provider.")
            provider = self.fallback

        # 2. Caching
        cache_key = self._get_cache_key(prompt, image_path)
        if self.cache:
            cached = await self.cache.get(cache_key)
            if cached:
                logger.info(f"VLM Cache Hit for key: {cache_key}")
                return cached
            logger.info(f"VLM Cache Miss for key: {cache_key}")

        # 3. RAG Context Injection
        final_prompt = prompt
        if use_rag and self.retriever:
            logger.info(f"Retrieving RAG context for prompt: {prompt[:50]}...")
            docs = await self.retriever.retrieve(prompt)
            if docs:
                context_parts = []
                for d in docs[:3]:
                    content = d.get("content", str(d)) if isinstance(d, dict) else str(d)
                    context_parts.append(content)
                context_str = "\n".join(context_parts)
                logger.info(f"Injected {len(docs)} documents into VLM prompt.")
                final_prompt = f"Context from database:\n{context_str}\n\nUser Question: {prompt}"
        
        if history:
            final_prompt = f"Conversation History:\n{history}\n\n{final_prompt}"
            
        if system_prompt:
            final_prompt = f"System: {system_prompt}\n\n{final_prompt}"

        # 4. Image Preprocessing
        processed_image = self._preprocess_image(image_path)

        # 5. VLM Generation
        try:
            response = await provider.generate_with_image(final_prompt, processed_image, session_id=session_id)
        except Exception as e:
            logger.error(f"VLM provider failed: {e}")
            if provider == self.primary:
                confirmed = await async_confirm(f"Primary VLM failed ({e}). Switch to Fallback for this request?")
                if confirmed:
                    if hasattr(self.fallback, "is_available") and not self.fallback.is_available():
                        return f"Error: Secondary provider is not configured. Cannot fallback."
                    
                    logger.info("Retrying with secondary provider...")
                    try:
                        response = await self.fallback.generate_with_image(final_prompt, processed_image, session_id=session_id)
                    except Exception as fallback_e:
                        return f"Error: Both primary and fallback providers failed.\nPrimary: {e}\nFallback: {fallback_e}"
                else:
                    return f"Error: Primary VLM failed and fallback was rejected. Details: {e}"
            else:
                raise e
        finally:
            # Cleanup temporary image if it was created
            if processed_image != image_path and os.path.exists(processed_image):
                try:
                    os.remove(processed_image)
                except Exception:
                    pass

        # 6. Save to Cache
        if self.cache:
            await self.cache.set(cache_key, response, ttl=3600)

        return response


