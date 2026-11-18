import ast
import asyncio
import json
import operator
import random
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from openai import AsyncOpenAI

from phoenix.core.config import config
from phoenix.core.container import container
from phoenix.services.llm.base import BaseLLM, _extract_json_dict
from phoenix.services.observability.logger import get_logger
from phoenix.services.observability.tracing import tracer

logger = get_logger("Phoenix AI.LLM.OpenAI")


def _extract_status_code(exc: Exception) -> Optional[int]:
    code = getattr(exc, "status_code", None)
    if isinstance(code, int):
        return code
    resp = getattr(exc, "response", None)
    if resp is not None and isinstance(getattr(resp, "status_code", None), int):
        return resp.status_code
    msg = str(exc)
    match = re.search(r"Error code:\s*(\d{3})", msg)
    if match:
        return int(match.group(1))
    return None


def _is_retryable_exception(exc: Exception) -> Tuple[bool, bool]:
    """
    Returns `(should_retry_same_model, is_rate_limit_or_unavailable)`.
    """
    code = _extract_status_code(exc)
    msg = str(exc).lower()
    if code == 429 or "rate limit" in msg or "too many requests" in msg:
        return True, True
    if code in (500, 502, 503, 504):
        return True, True
    if code in (401, 403, 404, 422):
        return False, True
    if code == 400:
        return False, False
    name = type(exc).__name__.lower()
    if any(k in name for k in ("timeout", "connection", "ratelimit", "internalserver", "serviceunavailable")):
        return True, True
    if any(k in msg for k in ("timeout", "timed out", "connection", "temporarily unavailable")):
        return True, True
    return True, False


def _is_tools_unsupported_error(exc: Exception) -> bool:
    code = _extract_status_code(exc)
    msg = str(exc).lower()
    if code in (400, 404, 422, 501) and any(
        kw in msg for kw in ("tool", "function", "tool_choice", "unsupported", "not support")
    ):
        return True
    return False


class OpenAILLM(BaseLLM):
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        fallback_models: Optional[List[str]] = None,
        max_retries: int = 2,
        retry_base_delay: float = 0.5,
        cooldown_seconds: float = 15.0,
    ):
        self.api_key = api_key or getattr(config, "OPENAI_API_KEY", None)
        self.base_url = base_url or getattr(config, "OPENAI_BASE_URL", None)
        self.model = model or getattr(config, "OPENAI_LLM_MODEL", None) or "LongCat-Flash-Chat"
        self.fallback_models: List[str] = list(fallback_models or [])
        self.max_retries = max(0, max_retries)
        self.retry_base_delay = max(0.05, retry_base_delay)
        self.cooldown_seconds = max(1.0, cooldown_seconds)
        self.client = None
        self._cooldowns: Dict[str, float] = {}
        self._tools_unsupported: set = set()

    @staticmethod
    def _safe_eval_math_expression(expression: str) -> Optional[float]:
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
        tool_outputs: List[str] = []
        calc_match = re.search(r"(?:^|\n)\s*calc\s*:\s*([0-9\.\+\-\*\/%\(\)\s\^]+)", prompt, flags=re.IGNORECASE)
        if calc_match:
            expr = calc_match.group(1).replace("^", "**").strip()
            calc_value = self._safe_eval_math_expression(expr)
            if calc_value is not None:
                val_str = str(int(calc_value)) if calc_value.is_integer() else str(calc_value)
                tool_outputs.append(f"Calculator result: {expr} = {val_str}")
        return tool_outputs

    async def _collect_memory_context(self, memory, session_id: str, prompt: str, smart_mode: bool) -> Dict[str, object]:
        history, history_context, semantic_context = [], "", ""
        if not session_id or not memory:
            return {"history": history, "history_context": history_context, "semantic_context": semantic_context}

        async def _safe(coro, default):
            try:
                return await coro
            except Exception:
                return default

        has_legacy = hasattr(memory, "history") and hasattr(memory, "get_context") and hasattr(memory, "search_memory")
        if not has_legacy:
            return {"history": history, "history_context": history_context, "semantic_context": semantic_context}

        history, history_context, semantic_context = await asyncio.gather(
            _safe(memory.history.get(session_id), []),
            _safe(memory.get_context(session_id), ""),
            _safe(memory.search_memory(session_id, prompt), ""),
        )
        if smart_mode and isinstance(history, list) and len(history) > 8:
            history = history[-8:]
        return {
            "history": history if isinstance(history, list) else [],
            "history_context": history_context if isinstance(history_context, str) else "",
            "semantic_context": semantic_context if isinstance(semantic_context, str) else "",
        }

    def _build_system_prompt(self, smart_mode: bool, semantic_context: str, history_context: str, tool_outputs: List[str]) -> str:
        base = "You are a helpful AI assistant."
        if smart_mode:
            base = (
                "You are a fast and accurate AI reasoning assistant.\n"
                "Priorities: Be correct, concise, and action-oriented."
            )
        segments = [base]
        if semantic_context:
            segments.append(f"Semantic context:\n{semantic_context}")
        if history_context:
            segments.append(f"Conversation summary:\n{history_context}")
        if tool_outputs:
            segments.append("Lightweight tool outputs:\n" + "\n\n".join(tool_outputs))
        return "\n\n".join(segments)

    def _ordered_candidate_models(self) -> List[str]:
        all_models = [self.model] + [m for m in self.fallback_models if m and m != self.model]
        now = time.time()
        healthy = [m for m in all_models if self._cooldowns.get(m, 0.0) <= now]
        cooled = [m for m in all_models if self._cooldowns.get(m, 0.0) > now]
        return healthy + cooled

    @staticmethod
    def _extract_usage(resp: Any) -> Dict[str, int]:
        usage_obj = getattr(resp, "usage", None)
        if not usage_obj:
            return {}
        return {
            "prompt_tokens": int(getattr(usage_obj, "prompt_tokens", 0) or 0),
            "completion_tokens": int(getattr(usage_obj, "completion_tokens", 0) or 0),
            "total_tokens": int(getattr(usage_obj, "total_tokens", 0) or 0),
        }

    async def _call_with_resilience(self, call_fn: Callable[[str], Any]) -> Tuple[Any, str]:
        """
        Execute an async API call across candidate models with exponential backoff + jitter
        and automatic model rotation/cooldown on 429 rate-limits or model errors.
        """
        if not self.client:
            await self.init()

        candidates = self._ordered_candidate_models()
        last_exc: Optional[Exception] = None

        for idx, model_name in enumerate(candidates):
            has_next = idx + 1 < len(candidates)
            # When fallback models exist, try once fast on rate-limit then fail over immediately
            attempts = (self.max_retries + 1) if not has_next else max(1, min(2, self.max_retries + 1))

            for attempt in range(attempts):
                try:
                    resp = await call_fn(model_name)
                    self._cooldowns.pop(model_name, None)
                    return resp, model_name
                except Exception as exc:
                    last_exc = exc
                    retryable, should_cool = _is_retryable_exception(exc)

                    if not retryable:
                        # Non-retryable (e.g. 401/404): break straight to next fallback model if available
                        break

                    if should_cool and _extract_status_code(exc) == 429:
                        self._cooldowns[model_name] = time.time() + self.cooldown_seconds
                        if has_next:
                            break

                    if attempt + 1 < attempts:
                        tracer.record_retry(model_name, str(exc))
                        delay = min(8.0, self.retry_base_delay * (2 ** attempt) + random.uniform(0.0, 0.25))
                        await asyncio.sleep(delay)

            if has_next and last_exc is not None:
                tracer.record_failover(model_name, candidates[idx + 1], str(last_exc))

        if last_exc is not None:
            raise last_exc
        raise RuntimeError("No models configured for OpenAILLM.")

    def is_available(self) -> bool:
        return bool(self.api_key) and bool(self.model)

    async def init(self):
        if not self.api_key:
            logger.warning("OPENAI_API_KEY is missing. Operating in mock mode.")
            self.client = None
            return

        base_url = self.base_url
        if base_url and base_url.endswith("/"):
            base_url = base_url[:-1]

        import httpx
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=base_url if base_url else None,
            timeout=httpx.Timeout(120.0, connect=60.0),
        )

    async def generate(
        self,
        prompt: str,
        session_id: Optional[str] = None,
        max_tokens: Optional[int] = None,
        mode: Optional[str] = None,
    ) -> str:
        if not self.api_key:
            return f"[Mock OpenAI Response (No API Key) to: {prompt}]"
        if not self.model:
            raise RuntimeError("OpenAILLM model is not configured.")
        if not self.client:
            await self.init()

        requested_mode = (mode or getattr(config, "OPENAI_GENERATION_MODE", "standard") or "standard").lower()
        smart_mode = requested_mode in {"smart", "fast_smart", "thinking", "planning"}

        memory = None
        current_messages = []
        try:
            memory = container.get("memory")
        except KeyError:
            pass

        tool_outputs = self._run_light_tools(prompt) if smart_mode else []
        mem_data = await self._collect_memory_context(memory, session_id, prompt, smart_mode)
        history = mem_data["history"]

        if session_id and memory:
            system_prompt = self._build_system_prompt(
                smart_mode=smart_mode,
                semantic_context=mem_data["semantic_context"],
                history_context=mem_data["history_context"],
                tool_outputs=tool_outputs,
            )
            current_messages.append({"role": "system", "content": system_prompt})
            for item in history:
                if isinstance(item, dict) and "content" in item:
                    current_messages.append(item["content"])
            current_messages.append({"role": "user", "content": prompt})
        else:
            if smart_mode:
                current_messages.append({
                    "role": "system",
                    "content": self._build_system_prompt(True, "", "", tool_outputs),
                })
            current_messages.append({"role": "user", "content": prompt})

        span_id = tracer.start_span("OpenAILLM.generate", {"model": self.model, "mode": requested_mode})
        try:
            async def _do_call(active_model: str):
                return await self.client.chat.completions.create(
                    model=active_model,
                    messages=current_messages,
                    max_tokens=max_tokens or config.SECURITY_MAX_OUTPUT_LENGTH,
                )

            resp, used_model = await self._call_with_resilience(_do_call)
            if not resp or not getattr(resp, "choices", None):
                tracer.end_span(span_id, status="error", error="Empty response from OpenAI")
                raise RuntimeError("Empty response from OpenAI API.")

            usage = self._extract_usage(resp)
            message = resp.choices[0].message
            content = message.content.strip() if message and message.content else ""

            if span_id in tracer.active_spans:
                tracer.active_spans[span_id]["metadata"]["model"] = used_model
            tracer.end_span(span_id, status="success", usage=usage)

            if session_id and memory and hasattr(memory, "add_interaction"):
                try:
                    await memory.add_interaction(session_id, prompt, content)
                except Exception:
                    pass

            return content
        except Exception as e:
            tracer.end_span(span_id, status="error", error=str(e))
            raise RuntimeError(f"OpenAILLM API call failed: {e}")

    async def generate_structured(
        self,
        prompt: str,
        schema: Any,
        session_id: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> Any:
        if not self.client:
            await self.init()

        schema_json = (
            schema.model_json_schema()
            if hasattr(schema, "model_json_schema")
            else (schema.schema_json() if hasattr(schema, "schema_json") else str(schema))
        )
        if isinstance(schema_json, dict):
            schema_json = json.dumps(schema_json, separators=(",", ":"))

        system_prompt = (
            f"You are a strict data-generation assistant. "
            f"You MUST output raw JSON matching the following JSON schema:\n{schema_json}\n"
            f"Do not output markdown code blocks or any conversational text. Just the raw valid JSON object."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        span_id = tracer.start_span("OpenAILLM.generate_structured", {"model": self.model})
        try:
            async def _do_call(active_model: str):
                try:
                    return await self.client.chat.completions.create(
                        model=active_model,
                        messages=messages,
                        max_tokens=max_tokens or config.SECURITY_MAX_OUTPUT_LENGTH,
                        response_format={"type": "json_object"},
                    )
                except Exception as inner_exc:
                    code = _extract_status_code(inner_exc)
                    # Retry without response_format only for provider capability rejection (400/422), NOT 429/timeout
                    if code in (400, 422):
                        return await self.client.chat.completions.create(
                            model=active_model,
                            messages=messages,
                            max_tokens=max_tokens or config.SECURITY_MAX_OUTPUT_LENGTH,
                        )
                    raise

            resp, used_model = await self._call_with_resilience(_do_call)
            content = (resp.choices[0].message.content or "").strip()
            usage = self._extract_usage(resp)
            if span_id in tracer.active_spans:
                tracer.active_spans[span_id]["metadata"]["model"] = used_model
            tracer.end_span(span_id, status="success", usage=usage)

            parsed_data = _extract_json_dict(content)
            if not parsed_data:
                parsed_data = json.loads(content)
            if hasattr(schema, "model_validate"):
                return schema.model_validate(parsed_data)
            return schema(**parsed_data)
        except Exception as e:
            tracer.end_span(span_id, status="error", error=str(e))
            raise RuntimeError(f"Structured generation failed: {e}")

    async def generate_with_tools(
        self,
        prompt: str,
        tools: List[Dict[str, Any]],
        session_id: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> Any:
        """
        Execute native OpenAI function calling (`tools=...`), returning `TaskExecutionSchema`.
        Automatically falls back to `generate_structured` if the provider rejects `tools`.
        """
        from phoenix.framework.agent.cognition.schema import TaskExecutionSchema, ToolCall

        if not tools or self.model in self._tools_unsupported:
            return await self.generate_structured(prompt, TaskExecutionSchema, session_id=session_id, max_tokens=max_tokens)

        if not self.client:
            await self.init()

        messages = [{"role": "user", "content": prompt}]
        span_id = tracer.start_span("OpenAILLM.generate_with_tools", {"model": self.model})

        try:
            async def _do_call(active_model: str):
                return await self.client.chat.completions.create(
                    model=active_model,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    max_tokens=max_tokens or config.SECURITY_MAX_OUTPUT_LENGTH,
                )

            resp, used_model = await self._call_with_resilience(_do_call)
            usage = self._extract_usage(resp)
            if span_id in tracer.active_spans:
                tracer.active_spans[span_id]["metadata"]["model"] = used_model
            tracer.end_span(span_id, status="success", usage=usage)

            msg = resp.choices[0].message
            raw_tool_calls = getattr(msg, "tool_calls", None) or []
            parsed_calls: List[ToolCall] = []

            for tc in raw_tool_calls:
                fn = getattr(tc, "function", None)
                if not fn:
                    continue
                t_name = getattr(fn, "name", "")
                raw_args = getattr(fn, "arguments", "{}") or "{}"
                if isinstance(raw_args, dict):
                    t_args = raw_args
                else:
                    try:
                        t_args = json.loads(raw_args)
                    except Exception:
                        t_args = _extract_json_dict(str(raw_args))
                if t_name:
                    parsed_calls.append(ToolCall(tool_name=t_name, arguments=t_args if isinstance(t_args, dict) else {}))

            content_text = (getattr(msg, "content", None) or "").strip()
            if parsed_calls:
                return TaskExecutionSchema(
                    thought_process=content_text or "Executing native tool calls.",
                    tools_to_call=parsed_calls,
                    is_complete=False,
                )

            return TaskExecutionSchema(
                thought_process=content_text,
                tools_to_call=[],
                direct_response=content_text,
                is_complete=True,
            )
        except Exception as e:
            tracer.end_span(span_id, status="error", error=str(e))
            if _is_tools_unsupported_error(e):
                self._tools_unsupported.add(self.model)
                return await self.generate_structured(
                    prompt, TaskExecutionSchema, session_id=session_id, max_tokens=max_tokens
                )
            raise RuntimeError(f"Native tool calling failed: {e}")
