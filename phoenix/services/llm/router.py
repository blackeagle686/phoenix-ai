import inspect
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

from phoenix.services.llm.base import BaseLLM
from phoenix.services.llm.openai import OpenAILLM
from phoenix.services.observability.tracing import tracer


class LLMRouter(BaseLLM):
    """
    Resilient Multi-LLM & Multi-Key Fallback Router.

    Supports both:
      1. Multiple platforms / API keys:
         LLMRouter([OpenAILLM(api_key=k1, ...), OpenAILLM(api_key=k2, ...)])
         or dicts: [{"api_key": k1, "base_url": u1, "model": m1}, ...]
      2. Single API key with multiple models (e.g. OpenRouter rate-limit failover):
         LLMRouter.from_models(api_key=k, base_url="https://openrouter.ai/api/v1", models=[m1, m2, m3])
    """

    def __init__(
        self,
        providers: Iterable[Union[BaseLLM, Dict[str, Any]]],
        cooldown_seconds: float = 20.0,
    ):
        self.providers: List[BaseLLM] = []
        for p in providers:
            if isinstance(p, dict):
                self.providers.append(OpenAILLM(**p))
            else:
                self.providers.append(p)

        if not self.providers:
            raise ValueError("LLMRouter requires at least one LLM provider.")

        self.cooldown_seconds = max(1.0, cooldown_seconds)
        self._cooldowns: Dict[int, float] = {}
        self.client = None  # Managed per provider or via init()
        self.last_used_provider: Optional[BaseLLM] = None

    @classmethod
    def from_models(
        cls,
        models: List[str],
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        cooldown_seconds: float = 20.0,
        **llm_kwargs: Any,
    ) -> "LLMRouter":
        """Create a router sharing one API key across multiple fallback models."""
        if not models:
            raise ValueError("models list must not be empty.")
        instances = [
            OpenAILLM(api_key=api_key, base_url=base_url, model=m, **llm_kwargs)
            for m in models
        ]
        return cls(instances, cooldown_seconds=cooldown_seconds)

    def _provider_label(self, idx: int, provider: BaseLLM) -> str:
        model = getattr(provider, "model", None) or getattr(provider, "model_path", None) or provider.__class__.__name__
        return f"{provider.__class__.__name__}({model})#{idx}"

    def _ordered_indices(self) -> List[int]:
        now = time.time()
        healthy = [i for i in range(len(self.providers)) if self._cooldowns.get(i, 0.0) <= now]
        cooled = [i for i in range(len(self.providers)) if self._cooldowns.get(i, 0.0) > now]
        return healthy + cooled

    def is_available(self) -> bool:
        return any(
            getattr(p, "is_available", lambda: True)()
            for p in self.providers
        )

    async def init(self):
        for p in self.providers:
            if hasattr(p, "init") and callable(p.init):
                res = p.init()
                if inspect.isawaitable(res):
                    await res
        self.client = True

    async def _route_call(self, fn_builder: Callable[[BaseLLM], Any]) -> Any:
        indices = self._ordered_indices()
        last_exc: Optional[Exception] = None

        for pos, idx in enumerate(indices):
            provider = self.providers[idx]
            label = self._provider_label(idx, provider)
            try:
                if hasattr(provider, "client") and getattr(provider, "client", None) is None and hasattr(provider, "init"):
                    res_init = provider.init()
                    if inspect.isawaitable(res_init):
                        await res_init

                result = await fn_builder(provider)
                self._cooldowns.pop(idx, None)
                self.last_used_provider = provider
                return result
            except Exception as exc:
                last_exc = exc
                msg = str(exc).lower()
                is_rl = "429" in msg or "rate limit" in msg or "too many requests" in msg
                self._cooldowns[idx] = time.time() + (self.cooldown_seconds if is_rl else min(5.0, self.cooldown_seconds))

                if pos + 1 < len(indices):
                    next_idx = indices[pos + 1]
                    next_label = self._provider_label(next_idx, self.providers[next_idx])
                    tracer.record_failover(label, next_label, str(exc))

        if last_exc is not None:
            raise last_exc
        raise RuntimeError("All providers in LLMRouter failed.")

    async def generate(
        self,
        prompt: str,
        session_id: Optional[str] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        async def _call(p: BaseLLM):
            sig = inspect.signature(p.generate)
            call_kw: Dict[str, Any] = {}
            if "session_id" in sig.parameters:
                call_kw["session_id"] = session_id
            if "max_tokens" in sig.parameters and max_tokens is not None:
                call_kw["max_tokens"] = max_tokens
            if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values()):
                call_kw.update(kwargs)
            return await p.generate(prompt, **call_kw)

        return await self._route_call(_call)

    async def generate_structured(
        self,
        prompt: str,
        schema: Any,
        session_id: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> Any:
        async def _call(p: BaseLLM):
            if hasattr(p, "generate_structured"):
                return await p.generate_structured(prompt, schema, session_id=session_id, max_tokens=max_tokens)
            return await super(LLMRouter, self).generate_structured(prompt, schema, session_id=session_id, max_tokens=max_tokens)

        return await self._route_call(_call)

    async def generate_with_tools(
        self,
        prompt: str,
        tools: List[Dict[str, Any]],
        session_id: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> Any:
        async def _call(p: BaseLLM):
            if hasattr(p, "generate_with_tools"):
                return await p.generate_with_tools(prompt, tools, session_id=session_id, max_tokens=max_tokens)
            return await p.generate_structured(
                prompt,
                __import__("phoenix.framework.agent.cognition.schema", fromlist=["TaskExecutionSchema"]).TaskExecutionSchema,
                session_id=session_id,
                max_tokens=max_tokens,
            )

        return await self._route_call(_call)

    async def generate_stream(self, prompt: str, session_id: Optional[str] = None, max_tokens: Optional[int] = None):
        indices = self._ordered_indices()
        last_exc: Optional[Exception] = None
        for pos, idx in enumerate(indices):
            provider = self.providers[idx]
            try:
                if hasattr(provider, "generate_stream"):
                    stream_iter = provider.generate_stream(prompt, session_id=session_id, max_tokens=max_tokens)
                    if hasattr(stream_iter, "__aiter__"):
                        async for chunk in stream_iter:
                            yield chunk
                        self.last_used_provider = provider
                        return
                text = await self.generate(prompt, session_id=session_id, max_tokens=max_tokens)
                yield text
                return
            except Exception as exc:
                last_exc = exc
                self._cooldowns[idx] = time.time() + self.cooldown_seconds
                if pos + 1 < len(indices):
                    next_idx = indices[pos + 1]
                    tracer.record_failover(
                        self._provider_label(idx, provider),
                        self._provider_label(next_idx, self.providers[next_idx]),
                        str(exc),
                    )
        if last_exc is not None:
            raise last_exc


MultiLLM = LLMRouter
FallbackLLM = LLMRouter

from phoenix.services.llm.moe import MoERouter, ExpertProfile, moe_route_context

__all__ = [
    "LLMRouter",
    "MultiLLM",
    "FallbackLLM",
    "MoERouter",
    "ExpertProfile",
    "moe_route_context",
]
