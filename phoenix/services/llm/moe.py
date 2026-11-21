"""Zero-overhead Mixture-of-Experts (MoE) Router for Phoenix LLM Services.

Routes calls deterministically across specialized LLM experts by agent cognitive role
(planner / coder / watcher / fast / synthesis), lightweight prompt domain hints, and
automatic retry escalation — with per-expert cooldown failover and zero extra gate tokens.
"""

from __future__ import annotations

import contextlib
import contextvars
import inspect
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

from phoenix.services.llm.base import BaseLLM
from phoenix.services.llm.openai import OpenAILLM
from phoenix.services.observability.tracing import tracer

_MOE_ROUTE_CTX: contextvars.ContextVar[Optional[Dict[str, Any]]] = contextvars.ContextVar(
    "phoenix_moe_route_ctx",
    default=None,
)

_CODE_HINT_RE = re.compile(
    r"\b(def |class |import |async def|pytest|unittest|refactor|bug|stacktrace|traceback|sql|select |file_edit|file_write|\.py\b|\.ts\b|\.js\b)",
    re.IGNORECASE,
)
_ARABIC_RE = re.compile(r"[\u0600-\u06FF]")

_NAME_DEFAULTS: Dict[str, Tuple[Tuple[str, ...], int, int]] = {
    "planner": (("planner", "deep_plan", "replan", "analyzer"), 2, 10),
    "architect": (("planner", "deep_plan", "replan", "analyzer"), 2, 10),
    "reasoning": (("planner", "watcher", "reflector", "replan"), 2, 10),
    "coder": (("coder", "tools", "solve_task", "react"), 1, 10),
    "tools": (("coder", "tools", "solve_task", "react"), 1, 10),
    "executor": (("coder", "tools", "solve_task", "react"), 1, 10),
    "watcher": (("watcher", "reflector"), 2, 10),
    "reviewer": (("watcher", "reflector"), 2, 10),
    "reflector": (("watcher", "reflector"), 2, 10),
    "critic": (("watcher", "reflector"), 2, 10),
    "fast": (("fast", "synthesis", "classifier"), 0, 8),
    "chat": (("fast", "synthesis", "classifier"), 0, 8),
    "router": (("fast", "classifier"), 0, 8),
    "synthesis": (("synthesis", "fast"), 0, 8),
    "synthesizer": (("synthesis", "fast"), 0, 8),
    "arabic": (("arabic", "fast", "synthesis"), 1, 9),
}


@contextlib.contextmanager
def moe_route_context(
    role: Optional[str] = None,
    *,
    escalate: bool = False,
    attempt: int = 1,
    requires_tools: Optional[bool] = None,
    hints: Optional[Dict[str, Any]] = None,
):
    """Attach lightweight routing metadata for downstream MoERouter calls in this async task."""
    prev = _MOE_ROUTE_CTX.get()
    merged: Dict[str, Any] = dict(prev or {})
    if role is not None:
        merged["role"] = role
    if escalate:
        merged["escalate"] = True
    if attempt != 1:
        merged["attempt"] = attempt
    if requires_tools is not None:
        merged["requires_tools"] = requires_tools
    if hints:
        merged.update(hints)
    token = _MOE_ROUTE_CTX.set(merged)
    try:
        yield merged
    finally:
        _MOE_ROUTE_CTX.reset(token)


def get_moe_route_context() -> Dict[str, Any]:
    return dict(_MOE_ROUTE_CTX.get() or {})


@dataclass
class ExpertProfile:
    """Configuration wrapper describing an LLM expert in MoERouter."""

    name: str
    llm: BaseLLM
    roles: Tuple[str, ...] = ()
    keywords: Tuple[str, ...] = ()
    priority: int = 0
    tier: int = 1
    supports_tools: bool = True
    cooldown_seconds: float = 20.0


def _build_llm_instance(spec: Any) -> BaseLLM:
    if isinstance(spec, dict):
        if "llm" in spec:
            inner = spec["llm"]
            return OpenAILLM(**inner) if isinstance(inner, dict) else inner
        return OpenAILLM(**spec)
    return spec


class MoERouter(BaseLLM):
    """
    Role-, Domain-, and Escalation-aware Mixture-of-Experts LLM Router.

    Accepts either:
      1) Dict of named experts:
         MoERouter({"planner": llm_reasoner, "coder": llm_coder, "watcher": llm_critic, "fast": llm_small})
      2) List of `ExpertProfile` objects.
      3) Dict entries with custom metadata:
         MoERouter({"sql_guru": {"llm": openai_llm, "roles": ["coder"], "keywords": ["sql", "database"], "tier": 2}})
    """

    def __init__(
        self,
        experts: Union[Dict[str, Any], Iterable[ExpertProfile]],
        default_expert: Optional[str] = None,
        escalate_on_failure: bool = True,
        cooldown_seconds: float = 20.0,
        gate_fn: Optional[Callable[[str, Dict[str, Any], List[ExpertProfile]], Optional[str]]] = None,
    ):
        self.experts: List[ExpertProfile] = []
        self._by_name: Dict[str, ExpertProfile] = {}

        if isinstance(experts, dict):
            for name, spec in experts.items():
                norm_name = str(name).strip()
                def_roles, def_tier, def_prio = _NAME_DEFAULTS.get(norm_name.lower(), ((norm_name.lower(), "general"), 1, 5))
                if isinstance(spec, ExpertProfile):
                    prof = spec
                elif isinstance(spec, dict) and any(k in spec for k in ("llm", "roles", "keywords", "tier", "priority", "supports_tools")):
                    prof = ExpertProfile(
                        name=norm_name,
                        llm=_build_llm_instance(spec.get("llm") or spec),
                        roles=tuple(spec.get("roles") or def_roles),
                        keywords=tuple(str(k).lower() for k in (spec.get("keywords") or ())),
                        priority=int(spec.get("priority", def_prio)),
                        tier=int(spec.get("tier", def_tier)),
                        supports_tools=bool(spec.get("supports_tools", True)),
                        cooldown_seconds=float(spec.get("cooldown_seconds", cooldown_seconds)),
                    )
                else:
                    prof = ExpertProfile(
                        name=norm_name,
                        llm=_build_llm_instance(spec),
                        roles=def_roles,
                        priority=def_prio,
                        tier=def_tier,
                        cooldown_seconds=float(cooldown_seconds),
                    )
                self._register_expert(prof)
        else:
            for item in experts:
                if not isinstance(item, ExpertProfile):
                    raise TypeError("Iterable passed to MoERouter must contain ExpertProfile instances.")
                self._register_expert(item)

        if not self.experts:
            raise ValueError("MoERouter requires at least one expert.")

        self.default_expert = default_expert if (default_expert and default_expert in self._by_name) else self.experts[0].name
        self.escalate_on_failure = bool(escalate_on_failure)
        self.cooldown_seconds = max(1.0, float(cooldown_seconds))
        self.gate_fn = gate_fn

        self._cooldowns: Dict[str, float] = {}
        self.client: Any = True
        self.last_used_expert: Optional[str] = None
        self.last_used_provider: Optional[BaseLLM] = None
        self.routing_stats: Dict[str, Any] = {
            "total_calls": 0,
            "calls_by_expert": {},
            "calls_by_role": {},
            "escalations": 0,
            "failovers": 0,
        }

    def _register_expert(self, profile: ExpertProfile) -> None:
        if profile.name in self._by_name:
            raise ValueError(f"Duplicate expert name in MoERouter: '{profile.name}'.")
        self.experts.append(profile)
        self._by_name[profile.name] = profile

    def get_routing_stats(self) -> Dict[str, Any]:
        return {
            "total_calls": self.routing_stats["total_calls"],
            "calls_by_expert": dict(self.routing_stats["calls_by_expert"]),
            "calls_by_role": dict(self.routing_stats["calls_by_role"]),
            "escalations": self.routing_stats["escalations"],
            "failovers": self.routing_stats["failovers"],
            "last_used_expert": self.last_used_expert,
        }

    def is_available(self) -> bool:
        return any(getattr(exp.llm, "is_available", lambda: True)() for exp in self.experts)

    async def init(self):
        """Mark MoERouter ready; experts initialize lazily on first routed call to conserve GPU VRAM."""
        self.client = True

    async def init_all(self):
        """Eagerly initialize every expert (useful on API keys or when multi-model caching fits in RAM)."""
        for exp in self.experts:
            if hasattr(exp.llm, "init") and callable(exp.llm.init):
                res = exp.llm.init()
                if inspect.isawaitable(res):
                    await res
        self.client = True

    def _infer_route(self, prompt: str, method: str, schema: Any = None) -> Tuple[str, Dict[str, Any]]:
        ctx = get_moe_route_context()
        schema_name = getattr(schema, "__name__", "") if schema is not None else ""
        explicit_role = bool(ctx.get("role"))
        role = ctx.get("role")

        if not role:
            if method == "generate_with_tools" or schema_name == "TaskExecutionSchema":
                role = "coder"
            elif schema_name == "PlanSchema":
                role = "planner"
            elif schema_name in ("ActionWatchSchema", "ReflectionSchema"):
                role = "watcher"
            elif "arabic" in self._by_name and _ARABIC_RE.search(prompt or "") and not _CODE_HINT_RE.search(prompt or ""):
                role = "arabic"
            elif _CODE_HINT_RE.search(prompt or ""):
                role = "coder"
            elif len(prompt or "") <= 180:
                role = "fast"
            else:
                role = "general"

        meta = {
            "role": role,
            "explicit_role": explicit_role,
            "method": method,
            "schema_name": schema_name,
            "escalate": bool(ctx.get("escalate") or int(ctx.get("attempt", 1) or 1) >= 2),
            "attempt": int(ctx.get("attempt", 1) or 1),
            "requires_tools": bool(ctx.get("requires_tools") or method == "generate_with_tools"),
        }
        return role, meta

    def _rank_experts(self, prompt: str, role: str, meta: Dict[str, Any]) -> List[ExpertProfile]:
        now = time.time()
        prompt_lower = (prompt or "").lower()
        custom_choice = self.gate_fn(prompt, meta, list(self.experts)) if self.gate_fn else None
        is_explicit_role = bool(meta.get("explicit_role"))

        scored: List[Tuple[Tuple[int, int, int], ExpertProfile]] = []
        for idx, exp in enumerate(self.experts):
            is_healthy = 1 if self._cooldowns.get(exp.name, 0.0) <= now else 0
            score = 0

            if custom_choice and exp.name == custom_choice:
                score += 500
            if meta.get("requires_tools") and exp.supports_tools:
                score += 30
            elif meta.get("requires_tools") and not exp.supports_tools:
                score -= 40

            if role in exp.roles:
                score += 140 if is_explicit_role else 90
            elif role == "synthesis" and "fast" in exp.roles:
                score += 75
            elif exp.name == self.default_expert:
                score += 15

            kw_hits = sum(1 for kw in exp.keywords if kw in prompt_lower)
            if kw_hits:
                score += min(45, kw_hits * 15) if is_explicit_role else (110 + min(45, (kw_hits - 1) * 15))

            if meta.get("escalate"):
                score += exp.tier * 35
            elif role in ("fast", "synthesis", "classifier"):
                score -= exp.tier * 5

            scored.append(((is_healthy, score + exp.priority, -idx), exp))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [item[1] for item in scored]

    async def _route_call(
        self,
        prompt: str,
        method: str,
        fn_builder: Callable[[BaseLLM], Any],
        schema: Any = None,
    ) -> Any:
        role, meta = self._infer_route(prompt, method, schema=schema)
        ranked = self._rank_experts(prompt, role, meta)
        last_exc: Optional[Exception] = None

        self.routing_stats["total_calls"] += 1
        self.routing_stats["calls_by_role"][role] = self.routing_stats["calls_by_role"].get(role, 0) + 1
        if meta.get("escalate"):
            self.routing_stats["escalations"] += 1

        for pos, exp in enumerate(ranked):
            try:
                if hasattr(exp.llm, "client") and getattr(exp.llm, "client", None) is None and hasattr(exp.llm, "init"):
                    res_init = exp.llm.init()
                    if inspect.isawaitable(res_init):
                        await res_init

                await tracer.emit_event(
                    "moe_route",
                    {
                        "expert": exp.name,
                        "role": role,
                        "tier": exp.tier,
                        "escalated": meta["escalate"],
                        "attempt": meta["attempt"],
                    },
                )
                result = await fn_builder(exp.llm)
                self._cooldowns.pop(exp.name, None)
                self.last_used_expert = exp.name
                self.last_used_provider = exp.llm
                self.routing_stats["calls_by_expert"][exp.name] = (
                    self.routing_stats["calls_by_expert"].get(exp.name, 0) + 1
                )
                return result
            except Exception as exc:
                last_exc = exc
                msg = str(exc).lower()
                is_rl = "429" in msg or "rate limit" in msg or "too many requests" in msg
                self._cooldowns[exp.name] = time.time() + (exp.cooldown_seconds if is_rl else min(5.0, exp.cooldown_seconds))

                if not self.escalate_on_failure:
                    raise
                if pos + 1 < len(ranked):
                    self.routing_stats["failovers"] += 1
                    next_exp = ranked[pos + 1]
                    tracer.record_failover(f"MoE({exp.name})", f"MoE({next_exp.name})", str(exc))

        if last_exc is not None:
            raise last_exc
        raise RuntimeError("All experts in MoERouter failed.")

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

        return await self._route_call(prompt, "generate", _call)

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
            return await super(MoERouter, self).generate_structured(prompt, schema, session_id=session_id, max_tokens=max_tokens)

        return await self._route_call(prompt, "generate_structured", _call, schema=schema)

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
            from phoenix.framework.agent.cognition.schema import TaskExecutionSchema

            return await p.generate_structured(prompt, TaskExecutionSchema, session_id=session_id, max_tokens=max_tokens)

        return await self._route_call(prompt, "generate_with_tools", _call)

    async def generate_stream(self, prompt: str, session_id: Optional[str] = None, max_tokens: Optional[int] = None):
        role, meta = self._infer_route(prompt, "generate_stream")
        ranked = self._rank_experts(prompt, role, meta)
        last_exc: Optional[Exception] = None
        for pos, exp in enumerate(ranked):
            try:
                if hasattr(exp.llm, "client") and getattr(exp.llm, "client", None) is None and hasattr(exp.llm, "init"):
                    res_init = exp.llm.init()
                    if inspect.isawaitable(res_init):
                        await res_init

                if hasattr(exp.llm, "generate_stream"):
                    stream_iter = exp.llm.generate_stream(prompt, session_id=session_id, max_tokens=max_tokens)
                    if hasattr(stream_iter, "__aiter__"):
                        async for chunk in stream_iter:
                            yield chunk
                        self.last_used_expert = exp.name
                        self.last_used_provider = exp.llm
                        return
                text = await exp.llm.generate(prompt, session_id=session_id, max_tokens=max_tokens)
                yield text
                return
            except Exception as exc:
                last_exc = exc
                self._cooldowns[exp.name] = time.time() + exp.cooldown_seconds
                if pos + 1 < len(ranked):
                    tracer.record_failover(f"MoE({exp.name})", f"MoE({ranked[pos + 1].name})", str(exc))
        if last_exc is not None:
            raise last_exc
