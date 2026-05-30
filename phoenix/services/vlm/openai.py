import base64
from phoenix.services.llm.base import BaseVLM
from phoenix.core.config import config
from openai import AsyncOpenAI
import os
import mimetypes

from phoenix.services.observability.tracing import tracer
from phoenix.core.container import container
from typing import Optional

class OpenAIVLM(BaseVLM):
    def __init__(self):
        self.api_key = getattr(config, "OPENAI_VLM_API_KEY", "") or getattr(config, "OPENAI_API_KEY", "")
        self.base_url = getattr(config, "OPENAI_VLM_BASE_URL", "") or getattr(config, "OPENAI_BASE_URL", "")
        self.model = getattr(config, "OPENAI_VLM_MODEL", "")
        self.client = None

    def is_available(self) -> bool:
        return bool(self.api_key) and not self.api_key.startswith("ak_") and bool(self.model)

    async def init(self):
        if not self.api_key:
            print("Warning: OPENAI_API_KEY is missing. Operating in mock mode.")
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def _get_mime_type(self, image_path: str) -> str:
        mime_type, _ = mimetypes.guess_type(image_path)
        return mime_type or "image/jpeg"

    async def generate_with_image(self, prompt: str, image_path: str, session_id: Optional[str] = None) -> str:
        if not self.client:
            raise RuntimeError("OpenAIVLM is not initialized.")

        # Handle Memory
        memory = None
        current_messages = []
        try:
            memory = container.get("memory")
        except KeyError:
            pass

        if session_id and memory:
            # 1. Retrieve history
            history_context = await memory.get_context(session_id)
            # 2. Retrieve semantic context
            semantic_context = await memory.search_memory(session_id, prompt)
            
            system_prompt = "You are a helpful Vision-AI assistant."
            if semantic_context:
                system_prompt += f"\n\nContext from previous conversations:\n{semantic_context}"
            
            current_messages.append({"role": "system", "content": system_prompt})
            
            history = await memory.history.get(session_id)
            for item in history:
                current_messages.append(item["content"])
        
        # Load and encode image
        import base64
        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

        current_messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"},
                },
            ],
        })

        if not self.api_key:
            return f"[Mock OpenAI VLM Response to: {prompt} with image {image_path}]"

        span_id = tracer.start_span("OpenAIVLM.generate", {"model": self.model})
        try:
            resp = await self.client.chat.completions.create(
                model=self.model,
                messages=current_messages,
                max_tokens=config.SECURITY_MAX_OUTPUT_LENGTH,
            )

            if not resp or not resp.choices:
                tracer.end_span(span_id, status="error", error="Empty response from OpenAI VLM")
                raise RuntimeError("Empty response from OpenAI VLM API.")

            message = resp.choices[0].message
            content = message.content.strip() if message and message.content else ""
            
            tracer.end_span(span_id, status="success")
            
            # Store interaction in memory
            if session_id and memory:
                await memory.add_interaction(session_id, prompt, content)
                
            return content

        except Exception as e:
            tracer.end_span(span_id, status="error", error=str(e))
            raise RuntimeError(f"OpenAIVLM API call failed: {e}")

