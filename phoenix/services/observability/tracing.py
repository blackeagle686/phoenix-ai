import contextvars
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from phoenix.services.observability.logger import get_logger

logger = get_logger("Phoenix AI.Telemetry")


@dataclass
class RunMetrics:
    """Per-run execution telemetry for tokens, latency, retries, failovers, and tools."""

    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    session_id: str = ""
    start_time: float = field(default_factory=time.perf_counter)
    duration_ms: float = 0.0
    status: str = "running"

    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    models_used: List[str] = field(default_factory=list)

    retries: int = 0
    failovers: int = 0
    failover_events: List[Dict[str, str]] = field(default_factory=list)

    tool_calls: int = 0
    tool_failures: int = 0
    tool_durations_ms: Dict[str, float] = field(default_factory=dict)

    def add_llm_usage(self, model: Optional[str] = None, usage: Optional[Dict[str, Any]] = None) -> None:
        self.llm_calls += 1
        if model and model not in self.models_used:
            self.models_used.append(model)
        if isinstance(usage, dict):
            self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
            self.completion_tokens += int(usage.get("completion_tokens") or 0)
            tot = usage.get("total_tokens")
            if tot is None:
                tot = int(usage.get("prompt_tokens") or 0) + int(usage.get("completion_tokens") or 0)
            self.total_tokens += int(tot or 0)

    def add_retry(self) -> None:
        self.retries += 1

    def add_failover(self, from_target: str, to_target: str, reason: str = "") -> None:
        self.failovers += 1
        self.failover_events.append({
            "from": from_target,
            "to": to_target,
            "reason": str(reason)[:200],
        })

    def add_tool_call(self, tool_name: str, duration_ms: float, success: bool) -> None:
        self.tool_calls += 1
        if not success:
            self.tool_failures += 1
        self.tool_durations_ms[tool_name] = round(
            self.tool_durations_ms.get(tool_name, 0.0) + duration_ms, 2
        )

    def finish(self, status: str = "success") -> Dict[str, Any]:
        self.duration_ms = round((time.perf_counter() - self.start_time) * 1000, 2)
        self.status = status
        return self.to_dict()

    def to_dict(self) -> Dict[str, Any]:
        elapsed = self.duration_ms if self.status != "running" else round((time.perf_counter() - self.start_time) * 1000, 2)
        return {
            "run_id": self.run_id,
            "session_id": self.session_id,
            "status": self.status,
            "duration_ms": elapsed,
            "llm_calls": self.llm_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "models_used": list(self.models_used),
            "retries": self.retries,
            "failovers": self.failovers,
            "failover_events": list(self.failover_events),
            "tool_calls": self.tool_calls,
            "tool_failures": self.tool_failures,
            "tool_durations_ms": dict(self.tool_durations_ms),
        }


_ACTIVE_RUN_METRICS: contextvars.ContextVar[Optional[RunMetrics]] = contextvars.ContextVar(
    "phoenix_active_run_metrics", default=None
)
_ACTIVE_EVENT_CALLBACK: contextvars.ContextVar[Optional[Callable[[Dict[str, Any]], Any]]] = contextvars.ContextVar(
    "phoenix_active_event_callback", default=None
)


class Telemetry:
    def __init__(self):
        self.active_spans: Dict[str, Dict[str, Any]] = {}

    def begin_run(
        self,
        session_id: str = "",
        on_event: Optional[Callable[[Dict[str, Any]], Any]] = None,
    ) -> RunMetrics:
        metrics = RunMetrics(session_id=session_id)
        _ACTIVE_RUN_METRICS.set(metrics)
        if on_event is not None:
            _ACTIVE_EVENT_CALLBACK.set(on_event)
        return metrics

    def get_active_run(self) -> Optional[RunMetrics]:
        return _ACTIVE_RUN_METRICS.get()

    def end_run(self, status: str = "success") -> Optional[Dict[str, Any]]:
        metrics = _ACTIVE_RUN_METRICS.get()
        _ACTIVE_RUN_METRICS.set(None)
        _ACTIVE_EVENT_CALLBACK.set(None)
        if metrics is not None:
            return metrics.finish(status=status)
        return None

    async def emit_event(self, event_type: str, data: Optional[Dict[str, Any]] = None) -> None:
        cb = _ACTIVE_EVENT_CALLBACK.get()
        if cb is None:
            return
        payload = {"type": event_type, "timestamp": time.time(), **(data or {})}
        try:
            import inspect
            res = cb(payload)
            if inspect.isawaitable(res):
                await res
        except Exception:
            pass

    def record_retry(self, target: str = "", reason: str = "") -> None:
        run = _ACTIVE_RUN_METRICS.get()
        if run is not None:
            run.add_retry()

    def record_failover(self, from_target: str, to_target: str, reason: str = "") -> None:
        run = _ACTIVE_RUN_METRICS.get()
        if run is not None:
            run.add_failover(from_target, to_target, reason)
        logger.warning(f"LLM Failover: {from_target} -> {to_target} ({str(reason)[:160]})")

    def record_tool(self, tool_name: str, duration_ms: float, success: bool) -> None:
        run = _ACTIVE_RUN_METRICS.get()
        if run is not None:
            run.add_tool_call(tool_name, duration_ms, success)

    def start_span(self, name: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        span_id = str(uuid.uuid4())
        self.active_spans[span_id] = {
            "name": name,
            "start_time": time.perf_counter(),
            "metadata": dict(metadata or {}),
        }
        return span_id

    def end_span(
        self,
        span_id: str,
        status: str = "success",
        error: Optional[str] = None,
        usage: Optional[Dict[str, int]] = None,
    ):
        if span_id not in self.active_spans:
            return

        span = self.active_spans.pop(span_id)
        duration = (time.perf_counter() - span["start_time"]) * 1000

        metadata = span["metadata"]
        metadata.update({
            "status": status,
            "duration_ms": round(duration, 2),
            "usage": usage or {},
        })
        if error:
            metadata["error"] = error

        run = _ACTIVE_RUN_METRICS.get()
        if run is not None and span["name"].startswith(("OpenAILLM.", "LocalLLM.", "BaseLLM.", "LLMRouter.")):
            if status == "success":
                run.add_llm_usage(model=metadata.get("model"), usage=usage)

        message = f"Request Completed: {span['name']} | Status: {status} | Latency: {metadata['duration_ms']}ms"
        if usage:
            message += f" | Tokens: {usage.get('total_tokens', 0)}"

        if status == "error":
            logger.error(message, extra={"metadata": metadata})
        else:
            logger.info(message, extra={"metadata": metadata})


tracer = Telemetry()
