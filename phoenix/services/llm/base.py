from phoenix.core.base import BaseService
from typing import Optional, Any
import json
import re


def _extract_json_dict(raw_output: str) -> dict:
    if not raw_output or not isinstance(raw_output, str):
        return {}
    text = raw_output.strip()
    match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if match:
        text = match.group(1).strip()
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except Exception:
        # Try finding outermost JSON object
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                data = json.loads(text[start : end + 1])
                return data if isinstance(data, dict) else {}
            except Exception:
                pass
        return {}


class BaseLLM(BaseService):
    def is_available(self) -> bool:
        raise NotImplementedError

    async def generate(self, prompt: str, session_id: Optional[str] = None, max_tokens: Optional[int] = None) -> str:
        raise NotImplementedError

    async def generate_stream(self, prompt: str, session_id: Optional[str] = None, max_tokens: Optional[int] = None):
        raise NotImplementedError

    async def generate_structured(
        self,
        prompt: str,
        schema: Any,
        session_id: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> Any:
        """
        Default structured output generation for any BaseLLM implementation.
        Prompts the model for strict JSON matching `schema` and validates it.
        """
        schema_json = (
            schema.model_json_schema()
            if hasattr(schema, "model_json_schema")
            else (schema.schema_json() if hasattr(schema, "schema_json") else str(schema))
        )
        if isinstance(schema_json, dict):
            schema_json = json.dumps(schema_json)

        structured_prompt = (
            f"{prompt}\n\n"
            f"IMPORTANT: Respond ONLY with a valid raw JSON object matching this JSON schema:\n"
            f"{schema_json}\n"
            f"Do not include markdown fences or extra text."
        )
        raw = await self.generate(structured_prompt, session_id=session_id, max_tokens=max_tokens)
        parsed = _extract_json_dict(raw)
        if hasattr(schema, "model_validate"):
            return schema.model_validate(parsed)
        return schema(**parsed)

    async def generate_with_tools(
        self,
        prompt: str,
        tools: Any,
        session_id: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> Any:
        """
        Default tool-calling fallback using `generate_structured` against `TaskExecutionSchema`.
        Subclasses supporting native API function calling override this method.
        """
        from phoenix.framework.agent.cognition.schema import TaskExecutionSchema

        return await self.generate_structured(
            prompt,
            TaskExecutionSchema,
            session_id=session_id,
            max_tokens=max_tokens,
        )


class BaseVLM(BaseService):
    def is_available(self) -> bool:
        raise NotImplementedError

    async def generate_with_image(self, prompt: str, image_path: str, session_id: Optional[str] = None) -> str:
        raise NotImplementedError

    async def generate_stream(self, prompt: str, session_id: Optional[str] = None):
        raise NotImplementedError


