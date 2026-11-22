from phoenix.core.base import BaseService
from typing import Optional, Any
import json
import re


def _repair_truncated_json(candidate: str) -> dict:
    """Recover a valid dict prefix when a small local model hits max_new_tokens mid-JSON."""
    start = candidate.find("{")
    if start == -1:
        return {}
    s = candidate[start:].strip()
    if not s:
        return {}

    def _close_json(text: str) -> str:
        in_str = False
        esc = False
        stack = []
        for ch in text:
            if esc:
                esc = False
                continue
            if ch == "\\" and in_str:
                esc = True
                continue
            if ch == '"':
                in_str = not in_str
            elif not in_str:
                if ch == "{":
                    stack.append("}")
                elif ch == "[":
                    stack.append("]")
                elif ch in ("}", "]") and stack and stack[-1] == ch:
                    stack.pop()
        out = text
        if in_str:
            if out.endswith("\\"):
                out = out[:-1]
            out += '"'
        out = re.sub(r",\s*$", "", out.rstrip())
        if re.search(r':\s*$', out):
            out += "null"
        out += "".join(reversed(stack))
        return re.sub(r",\s*([}\]])", r"\1", out)

    try:
        data = json.loads(_close_json(s), strict=False)
        if isinstance(data, dict) and data:
            return data
    except Exception:
        pass
    return {}


def _try_parse_json_dict(candidate: str) -> dict:
    for strict_mode in (True, False):
        try:
            data = json.loads(candidate, strict=strict_mode)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    cleaned = re.sub(r",\s*([}\]])", r"\1", candidate)
    for strict_mode in (True, False):
        try:
            data = json.loads(cleaned, strict=strict_mode)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return {}


def _extract_json_dict(raw_output: str) -> dict:
    if not raw_output or not isinstance(raw_output, str):
        return {}
    # Strip reasoning model `<think>...</think>` blocks cleanly before JSON extraction (e.g., Qwen3 / DeepSeek-R1)
    text = re.sub(r"<think>.*?</think>", "", raw_output, flags=re.DOTALL).strip() or raw_output.strip()
    match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if match:
        parsed = _try_parse_json_dict(match.group(1).strip())
        if parsed:
            return parsed

    parsed = _try_parse_json_dict(text)
    if parsed:
        return parsed

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        parsed = _try_parse_json_dict(text[start : end + 1])
        if parsed:
            return parsed

    return _repair_truncated_json(text)


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


