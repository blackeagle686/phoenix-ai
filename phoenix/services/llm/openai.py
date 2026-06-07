from phoenix.services.llm.base import BaseLLM
from phoenix.core.config import config
from openai import AsyncOpenAI
from phoenix.services.observability.tracing import tracer
from phoenix.services.observability.logger import get_logger
from phoenix.core.container import container
from typing import Optional, List, Dict, Any
import asyncio
import re
import ast
import operator

logger = get_logger("Phoenix AI.LLM.OpenAI")

class OpenAILLM(BaseLLM):
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or getattr(config, "OPENAI_API_KEY", None)
        self.base_url = base_url or getattr(config, "OPENAI_BASE_URL", None)
        self.model = model or getattr(config, "OPENAI_LLM_MODEL", None) or "LongCat-Flash-Chat"
        self.client = None

    @staticmethod
    def _safe_eval_math_expression(expression: str) -> Optional[float]:
        """
        Evaluate simple arithmetic expressions safely.
        Supported: +, -, *, /, %, **, unary +/- and parentheses.
        """
        allowed_ops = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Mod: operator.mod,
            ast.Pow: operator.pow,
            ast.USub: operator.neg,
            ast.UAdd: operator.pos,
        }

        def _eval(node):
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                return node.value
            if isinstance(node, ast.BinOp) and type(node.op) in allowed_ops:
                return allowed_ops[type(node.op)](_eval(node.left), _eval(node.right))
            if isinstance(node, ast.UnaryOp) and type(node.op) in allowed_ops:
                return allowed_ops[type(node.op)](_eval(node.operand))
            raise ValueError("Unsupported math expression.")

        try:
            parsed = ast.parse(expression, mode="eval")
            return float(_eval(parsed.body))
        except Exception:
            return None

    def _run_light_tools(self, prompt: str) -> List[str]:
        """
        Lightweight local helpers to speed up structured tasks.
        These tools do not call external services.
        """
        tool_outputs: List[str] = []

        # Calculator helper: detect "calc: <expression>"
        calc_match = re.search(r"(?:^|\n)\s*calc\s*:\s*([0-9\.\+\-\*\/%\(\)\s\^]+)", prompt, flags=re.IGNORECASE)
        if calc_match:
            expr = calc_match.group(1).replace("^", "**").strip()
            calc_value = self._safe_eval_math_expression(expr)
            if calc_value is not None:
                if calc_value.is_integer():
                    tool_outputs.append(f"Calculator result: {expr} = {int(calc_value)}")
                else:
                    tool_outputs.append(f"Calculator result: {expr} = {calc_value}")

        # Optimization helper: extract objectives/constraints from common keywords.
        lower = prompt.lower()
        has_opt_signal = any(k in lower for k in ["optimize", "optimization", "minimize", "maximize"])
        if has_opt_signal:
            constraints = re.findall(r"(?:constraint|limit|must|should)\s*:\s*([^\n]+)", prompt, flags=re.IGNORECASE)
            objective = re.search(r"(?:optimize|minimize|maximize)\s*[:\-]?\s*([^\n]+)", prompt, flags=re.IGNORECASE)

            if objective or constraints:
                objective_text = objective.group(1).strip() if objective else "Not explicitly provided"
                constraints_text = "; ".join(c.strip() for c in constraints[:5]) if constraints else "None explicit"
                tool_outputs.append(
                    "Optimization parser:\n"
                    f"- Objective: {objective_text}\n"
                    f"- Constraints: {constraints_text}\n"
                    "- Suggested strategy: solve constraints first, then optimize objective."
                )

        return tool_outputs

    async def _collect_memory_context(self, memory, session_id: str, prompt: str, smart_mode: bool) -> Dict[str, object]:
        """
        Collect memory in parallel for lower latency.
        """
        history = []
        history_context = ""
        semantic_context = ""

        if not session_id or not memory:
            return {
                "history": history,
                "history_context": history_context,
                "semantic_context": semantic_context
            }

        async def _safe(coro, default):
            try:
                return await coro
            except Exception:
                return default

        history_task = _safe(memory.history.get(session_id), [])
        history_context_task = _safe(memory.get_context(session_id), "")
        semantic_task = _safe(memory.search_memory(session_id, prompt), "")

        history, history_context, semantic_context = await asyncio.gather(
            history_task, history_context_task, semantic_task
        )

        # Keep only the latest turns in smart mode for faster token usage.
        if smart_mode and isinstance(history, list) and len(history) > 8:
            history = history[-8:]

        return {
            "history": history if isinstance(history, list) else [],
            "history_context": history_context if isinstance(history_context, str) else "",
            "semantic_context": semantic_context if isinstance(semantic_context, str) else ""
        }

    def _build_system_prompt(self, smart_mode: bool, semantic_context: str, history_context: str, tool_outputs: List[str]) -> str:
        base = "You are a helpful AI assistant."
        if smart_mode:
            base = (
                "You are a fast and accurate AI reasoning assistant.\n"
                "Priorities:\n"
                "1) Be correct.\n"
                "2) Be concise and action-oriented.\n"
                "3) Use planning internally, but return clear final answers.\n"
                "4) When tool outputs are provided, use them as trusted context."
            )

        segments = [base]

        if semantic_context:
            segments.append(f"Semantic context from previous conversations:\n{semantic_context}")
        if history_context:
            segments.append(f"Conversation summary context:\n{history_context}")
        if tool_outputs:
            segments.append("Lightweight tool outputs:\n" + "\n\n".join(tool_outputs))

        return "\n\n".join(segments)

    def is_available(self) -> bool:
        # LongCat keys often start with 'ak_', standard OpenAI keys start with 'sk-'
        return bool(self.api_key) and bool(self.model)
    
    async def init(self):
        if not self.api_key:
            logger.warning("OPENAI_API_KEY is missing. Operating in mock mode.")

        base_url = self.base_url
        if base_url and base_url.endswith("/"):
            base_url = base_url[:-1]

        import httpx
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=base_url if base_url else None,
            timeout=httpx.Timeout(120.0, connect=60.0)
        )

    async def generate(
        self,
        prompt: str,
        session_id: Optional[str] = None,
        max_tokens: Optional[int] = None,
        mode: Optional[str] = None
    ) -> str:
        if not self.client:
            raise RuntimeError("OpenAILLM is not initialized.")

        requested_mode = (mode or getattr(config, "OPENAI_GENERATION_MODE", "standard") or "standard").lower()
        smart_mode = requested_mode in {"smart", "fast_smart", "thinking", "planning"}

        # Handle memory
        memory = None
        current_messages = []

        try:
            memory = container.get("memory")
        except KeyError:
            pass

        # Run local helper tools only in smart mode to keep standard path minimal.
        tool_outputs = self._run_light_tools(prompt) if smart_mode else []

        mem_data = await self._collect_memory_context(memory, session_id, prompt, smart_mode)
        history = mem_data["history"]
        history_context = mem_data["history_context"]
        semantic_context = mem_data["semantic_context"]

        if session_id and memory:
            system_prompt = self._build_system_prompt(
                smart_mode=smart_mode,
                semantic_context=semantic_context,
                history_context=history_context,
                tool_outputs=tool_outputs
            )
            current_messages.append({"role": "system", "content": system_prompt})
            for item in history:
                if isinstance(item, dict) and "content" in item:
                    current_messages.append(item["content"])
        else:
            # For stateless smart mode, still provide smart system behavior.
            if smart_mode:
                current_messages.append({
                    "role": "system",
                    "content": self._build_system_prompt(
                        smart_mode=True,
                        semantic_context="",
                        history_context="",
                        tool_outputs=tool_outputs
                    )
                })
            current_messages.append({"role": "user", "content": prompt})

        # Mock mode
        if not self.api_key:
            return f"[Mock OpenAI Response (No API Key) to: {prompt}]"

        if not self.model:
            raise RuntimeError("OpenAILLM model is not configured.")

        span_id = tracer.start_span("OpenAILLM.generate", {"model": self.model, "mode": requested_mode})
        try:
            # Final user message if using memory
            if session_id and memory:
                current_messages.append({"role": "user", "content": prompt})

            resp = await self.client.chat.completions.create(
                model=self.model,
                messages=current_messages,
                max_tokens=max_tokens or config.SECURITY_MAX_OUTPUT_LENGTH
            )

            # Safe extraction
            if not resp or not resp.choices:
                tracer.end_span(span_id, status="error", error="Empty response from OpenAI")
                raise RuntimeError("Empty response from OpenAI API.")

            usage = {
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
                "total_tokens": resp.usage.total_tokens
            }
            
            message = resp.choices[0].message
            content = message.content.strip() if message and message.content else ""
            
            tracer.end_span(span_id, status="success", usage=usage)
            
            # Store interaction in memory
            if session_id and memory:
                await memory.add_interaction(session_id, prompt, content)
                
            return content

        except Exception as e:
            tracer.end_span(span_id, status="error", error=str(e))
            raise RuntimeError(f"OpenAILLM API call failed: {e}")

    async def generate_structured(
        self,
        prompt: str,
        schema: Any,
        session_id: Optional[str] = None,
        max_tokens: Optional[int] = None
    ) -> Any:
        if not self.client:
            raise RuntimeError("OpenAILLM is not initialized.")

        import json
        # Handle pydantic v1 vs v2 schema extraction
        schema_json = schema.model_json_schema() if hasattr(schema, "model_json_schema") else (schema.schema_json() if hasattr(schema, "schema_json") else str(schema))
        if isinstance(schema_json, dict):
            schema_json = json.dumps(schema_json)

        system_prompt = (
            f"You are a strict data-generation assistant. "
            f"You MUST output raw JSON matching the following JSON schema:\n{schema_json}\n"
            f"Do not output markdown code blocks or any conversational text. Just the raw valid JSON object."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]

        span_id = tracer.start_span("OpenAILLM.generate_structured", {"model": self.model})
        try:
            # We try using response_format, if it fails due to API compatibility, we could fallback, 
            # but LongCat supports json_object.
            resp = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens or config.SECURITY_MAX_OUTPUT_LENGTH,
                response_format={"type": "json_object"}
            )

            content = resp.choices[0].message.content.strip()
            tracer.end_span(span_id, status="success")

            # Remove potential markdown formatting if the model ignored instructions
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]

            parsed_data = json.loads(content.strip())
            return schema(**parsed_data)
        except Exception as e:
            tracer.end_span(span_id, status="error", error=str(e))
            raise RuntimeError(f"Structured generation failed: {e}")

