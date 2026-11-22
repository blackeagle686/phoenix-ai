"""CVAgent (MicroCVAgent): Ultra-Lightweight Side-by-Side Computer Vision Event Agent.

Designed to run on mobile phone CPUs and low-RAM edge devices alongside CV detectors
(YOLO, OpenCV, ONNX, TFLite, MediaPipe, DeepStream):
  - Strictly enforces MoE models <= 1.0B parameters via `MicroMoELLM`.
  - Fixed-RAM sliding temporal context window (`SlidingCVContext`).
  - Fast deterministic rules + <=1B MoE tiny decisions filling structured `DecisionGroup`s.
  - Fully user-modifiable tool registry (`send_alert`, `read_file`, `write_file`,
    local SQLite `db_connect`/`db_read_records`/`db_query`, plus custom CV triggers).
"""

from __future__ import annotations

import asyncio
import inspect
import json
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, AsyncIterable, Callable, Deque, Dict, List, Optional, Sequence, Union

_DC_SLOTS: Dict[str, Any] = {"slots": True} if sys.version_info >= (3, 10) else {}

from phoenix.framework.cv_agent.context import ContextSnapshot, SlidingCVContext
from phoenix.framework.cv_agent.decision import (
    ContextDecisionRule,
    DecisionGroup,
    MicroDecisionEngine,
    MicroToolCall,
    TinyDecision,
)
from phoenix.framework.cv_agent.events import CVEvent, CVSeverity
from phoenix.framework.cv_agent.moe_model import (
    HARD_MAX_MOE_PARAMS_B,
    MicroMoELLM,
    MicroMoEPolicyError,
)
from phoenix.framework.cv_agent.tools import (
    AlertDispatcher,
    LocalFileStore,
    LocalSQLiteStore,
    MicroTool,
    MicroToolRegistry,
    MicroToolResult,
    build_default_cv_tools,
)


@dataclass(**_DC_SLOTS)
class CVAgentConfig:
    """Compact runtime configuration for phone CPU / low-RAM CV deployments."""

    workspace_root: str = "./cv_workspace"
    max_context_events: int = 32
    context_window_seconds: float = 60.0
    alert_cooldown_seconds: float = 15.0
    max_db_rows: int = 50
    max_read_bytes: int = 65536
    max_write_bytes: int = 262144
    allow_db_writes: bool = True
    auto_send_alert_on_decision: bool = True
    auto_log_file: Optional[str] = None  # e.g. "cv_decisions.jsonl"
    policy_hint: str = ""
    context_slots: List[str] = field(default_factory=lambda: ["subject", "zone_status", "recommended_next"])


@dataclass(**_DC_SLOTS)
class CVAgentResult:
    """Complete output of handling one CVEvent."""

    event: CVEvent
    context: ContextSnapshot
    decision_group: DecisionGroup
    tool_results: List[MicroToolResult] = field(default_factory=list)
    alert_dispatched: bool = False
    used_moe: bool = False
    latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event": self.event.to_compact_dict(),
            "context": self.context.to_dict(),
            "decision_group": self.decision_group.to_dict(),
            "tool_results": [r.to_dict() for r in self.tool_results],
            "alert_dispatched": self.alert_dispatched,
            "used_moe": self.used_moe,
            "latency_ms": round(self.latency_ms, 3),
        }


class CVAgent:
    """
    Micron-footprint side-by-side Computer Vision Agent supporting ONLY <= 1B MoE models.

    Can operate in:
      1. Hybrid Rule + <=1B MoE mode (pass `model="granite-1b-moe"` or `MicroMoELLM(...)`)
      2. Zero-Model Ultra-Edge mode (`model=None`, deterministic rules + DB enrichment + alerts + files)
    """

    def __init__(
        self,
        model: Optional[Union[str, MicroMoELLM]] = "ibm-granite/granite-3.1-1b-a400m-instruct",
        *,
        config: Optional[CVAgentConfig] = None,
        workspace_root: Optional[str] = None,
        db_path: Optional[str] = None,
        db_alias: str = "default",
        on_alert: Optional[Callable[[Dict[str, Any]], Any]] = None,
        alert_log_path: Optional[str] = None,
        webhook_url: Optional[str] = None,
        rules: Optional[Sequence[ContextDecisionRule]] = None,
        db_enricher: Optional[Callable[[CVEvent, LocalSQLiteStore], List[Dict[str, Any]]]] = None,
        max_params_b: float = HARD_MAX_MOE_PARAMS_B,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.config = config or CVAgentConfig()
        if workspace_root is not None:
            self.config.workspace_root = workspace_root

        # Enforce strict <= 1B MoE model policy
        if model is None:
            self.llm: Optional[MicroMoELLM] = None
        elif isinstance(model, MicroMoELLM):
            if model.spec.params_b > HARD_MAX_MOE_PARAMS_B or model.spec.active_params_b > HARD_MAX_MOE_PARAMS_B:
                raise MicroMoEPolicyError(
                    f"MicroMoELLM '{model.model}' exceeds hard {HARD_MAX_MOE_PARAMS_B}B ceiling."
                )
            self.llm = model
        elif isinstance(model, str):
            kw = dict(model_kwargs or {})
            kw.setdefault("max_params_b", min(float(max_params_b), HARD_MAX_MOE_PARAMS_B))
            self.llm = MicroMoELLM(model=model, **kw)
        else:
            raise MicroMoEPolicyError(
                f"CVAgent only accepts a <=1B `MicroMoELLM`, valid <=1B MoE model string, or None; "
                f"got unsupported LLM type '{type(model).__name__}'."
            )

        self.context = SlidingCVContext(
            max_events=self.config.max_context_events,
            window_seconds=self.config.context_window_seconds,
            alert_cooldown_seconds=self.config.alert_cooldown_seconds,
        )
        self.files = LocalFileStore(
            workspace_root=self.config.workspace_root,
            max_read_bytes=self.config.max_read_bytes,
            max_write_bytes=self.config.max_write_bytes,
        )
        self.db = LocalSQLiteStore(
            max_rows=self.config.max_db_rows,
            allow_writes=self.config.allow_db_writes,
        )
        if db_path:
            self.db.connect(db_path, alias=db_alias)

        self.alerts = AlertDispatcher(
            on_alert=on_alert,
            alert_log_path=alert_log_path,
            webhook_url=webhook_url,
        )
        self.tools = MicroToolRegistry()
        build_default_cv_tools(
            registry=self.tools,
            alerts=self.alerts,
            files=self.files,
            db=self.db,
        )

        self._rules: List[ContextDecisionRule] = sorted(list(rules or []), key=lambda r: r.priority)
        self._db_enricher = db_enricher
        self._hooks: Dict[str, List[Callable[[CVAgentResult], Any]]] = {}

        # Compact bounded telemetry
        self._latencies_ms: Deque[float] = deque(maxlen=256)
        self._stats: Dict[str, int] = {
            "events_processed": 0,
            "rule_fast_hits": 0,
            "moe_calls": 0,
            "alerts_sent": 0,
            "tool_calls": 0,
            "tool_errors": 0,
        }

    # --- User Tool Modification API ---

    def tool(
        self,
        name: Optional[str] = None,
        *,
        description: Optional[str] = None,
        default_args: Optional[Dict[str, Any]] = None,
        tags: Sequence[str] = ("custom",),
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator to register or override a CV event handler tool."""
        def _decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            self.tools.register(
                fn,
                name=name or fn.__name__,
                description=description,
                default_args=default_args,
                tags=tags,
                overwrite=True,
            )
            return fn
        return _decorator

    def register_tool(
        self,
        tool_or_fn: Any,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        default_args: Optional[Dict[str, Any]] = None,
        enabled: bool = True,
        tags: Sequence[str] = ("custom",),
        overwrite: bool = True,
    ) -> MicroTool:
        return self.tools.register(
            tool_or_fn,
            name=name,
            description=description,
            default_args=default_args,
            enabled=enabled,
            tags=tags,
            overwrite=overwrite,
        )

    def modify_tool(
        self,
        name: str,
        *,
        fn: Optional[Callable[..., Any]] = None,
        description: Optional[str] = None,
        default_args: Optional[Dict[str, Any]] = None,
        merge_default_args: bool = True,
        enabled: Optional[bool] = None,
        tags: Optional[Sequence[str]] = None,
    ) -> MicroTool:
        """Modify any built-in or user tool (swap callable, tune default args, or toggle enabled)."""
        return self.tools.modify(
            name,
            fn=fn,
            description=description,
            default_args=default_args,
            merge_default_args=merge_default_args,
            enabled=enabled,
            tags=tags,
        )

    def remove_tool(self, name: str) -> bool:
        return self.tools.unregister(name)

    def list_tools(self) -> List[Dict[str, Any]]:
        return self.tools.list_tools()

    # --- Rules, Hooks, DB & File Direct Helpers ---

    def add_rule(self, rule: ContextDecisionRule) -> None:
        self._rules.append(rule)
        self._rules.sort(key=lambda r: r.priority)

    def set_db_enricher(self, fn: Optional[Callable[[CVEvent, LocalSQLiteStore], List[Dict[str, Any]]]]) -> None:
        """Attach a fast local SQLite lookup hook that enriches each CVEvent's context snapshot."""
        self._db_enricher = fn

    def on_event(self, event_type: str, callback: Callable[[CVAgentResult], Any]) -> None:
        """Subscribe a callback to a specific CV event type (or `'*'` for all events)."""
        self._hooks.setdefault(event_type, []).append(callback)

    def connect_database(self, db_path: str, *, alias: str = "default") -> Dict[str, Any]:
        return self.db.connect(db_path, alias=alias)

    def read_db_records(
        self,
        table: str,
        *,
        where: Optional[Dict[str, Any]] = None,
        columns: Optional[Sequence[str]] = None,
        limit: Optional[int] = None,
        order_by: Optional[str] = None,
        descending: bool = False,
        alias: str = "default",
    ) -> Dict[str, Any]:
        return self.db.read_records(
            table,
            where=where,
            columns=columns,
            limit=limit,
            order_by=order_by,
            descending=descending,
            alias=alias,
        )

    def read_file(self, path: str, *, max_bytes: Optional[int] = None) -> Dict[str, Any]:
        return self.files.read_file(path, max_bytes=max_bytes)

    def write_file(self, path: str, content: str, *, append: bool = False) -> Dict[str, Any]:
        return self.files.write_file(path, content, append=append)

    # --- Core Event Processing Pipeline ---

    async def handle_event(
        self,
        event: Union[CVEvent, Dict[str, Any]],
        *,
        force_moe: bool = False,
        extra_context: Optional[Dict[str, Any]] = None,
    ) -> CVAgentResult:
        """Ingest one CVEvent, fill a verified context DecisionGroup, execute micro tools & alerts."""
        t0 = time.perf_counter()
        cv_event = event if isinstance(event, CVEvent) else CVEvent.from_dict(event)
        if extra_context:
            cv_event.metadata.update(extra_context)

        # 1. Update bounded sliding temporal context (< 0.1ms)
        snapshot = self.context.record_event(cv_event)

        # 2. Optional local SQLite record lookup enrichment (known faces/plates/zone rules)
        if self._db_enricher is not None and self.db.connected_aliases():
            try:
                facts = self._db_enricher(cv_event, self.db)
                if inspect.isawaitable(facts):
                    facts = await facts
                if isinstance(facts, list):
                    snapshot.db_facts = facts[:8]
            except Exception:
                pass

        # 3. Evaluate fast deterministic ContextDecisionRules before waking <=1B MoE
        decision_group: Optional[DecisionGroup] = None
        used_moe = False
        if not force_moe:
            for rule in self._rules:
                try:
                    if rule.predicate(cv_event, snapshot):
                        built = rule.builder(cv_event, snapshot)
                        if rule.bypass_moe:
                            decision_group = built
                            if not decision_group.source:
                                decision_group.source = "rule_fast_path"
                            self._stats["rule_fast_hits"] += 1
                            break
                except Exception:
                    continue

        # 4. Run <= 1B MoE micro-inference if no fast-path rule bypassed it
        enabled_tools = self.tools.enabled_names()
        allowed_set = set(enabled_tools)
        if decision_group is None:
            if self.llm is not None:
                prompt = MicroDecisionEngine.build_micro_prompt(
                    cv_event,
                    snapshot,
                    available_tools=enabled_tools,
                    context_schema_keys=self.config.context_slots,
                    policy_hint=self.config.policy_hint,
                )
                raw_reply = await self.llm.generate(prompt)
                used_moe = True
                self._stats["moe_calls"] += 1
                decision_group = MicroDecisionEngine.parse_and_fill_decision_group(
                    raw_reply,
                    cv_event,
                    snapshot,
                    allowed_tools=allowed_set,
                    required_context_slots=self.config.context_slots,
                    source="micro_moe",
                )
            else:
                decision_group = MicroDecisionEngine.parse_and_fill_decision_group(
                    "",
                    cv_event,
                    snapshot,
                    allowed_tools=allowed_set,
                    required_context_slots=self.config.context_slots,
                    source="rule_fast_path",
                )

        # 5. Execute planned micro tool calls
        tool_results: List[MicroToolResult] = []
        alert_dispatched = False
        explicit_alert_called = False

        for tc in decision_group.tool_calls:
            self._stats["tool_calls"] += 1
            res = await self.tools.execute(tc.tool_name, tc.arguments)
            tool_results.append(res)
            if not res.success:
                self._stats["tool_errors"] += 1
            if tc.tool_name == "send_alert" and res.success:
                explicit_alert_called = True
                alert_dispatched = True
                self._stats["alerts_sent"] += 1
                self.context.mark_alert_sent(cv_event, ts=cv_event.timestamp)

        # 6. Auto-dispatch alert via `send_alert` tool if decision_group requires it & wasn't already sent
        if (
            self.config.auto_send_alert_on_decision
            and decision_group.alert_needed
            and not explicit_alert_called
            and "send_alert" in allowed_set
        ):
            self._stats["tool_calls"] += 1
            alert_args = {
                "message": decision_group.alert_message or decision_group.summary,
                "severity": decision_group.risk_level.value,
                "camera_id": cv_event.camera_id,
                "zone": cv_event.zone,
                "event_id": cv_event.event_id,
                "metadata": dict(decision_group.context_slots),
            }
            a_res = await self.tools.execute("send_alert", alert_args)
            tool_results.append(a_res)
            if a_res.success:
                alert_dispatched = True
                self._stats["alerts_sent"] += 1
                self.context.mark_alert_sent(cv_event, ts=cv_event.timestamp)
            else:
                self._stats["tool_errors"] += 1

        # 7. Optional auto JSONL audit log inside workspace_root
        if self.config.auto_log_file and "append_file" in allowed_set:
            log_line = json.dumps(
                {
                    "ts": round(cv_event.timestamp, 3),
                    "event_id": cv_event.event_id,
                    "event_type": cv_event.event_type,
                    "camera_id": cv_event.camera_id,
                    "zone": cv_event.zone,
                    "risk": decision_group.risk_level.value,
                    "summary": decision_group.summary,
                    "alert_sent": alert_dispatched,
                },
                ensure_ascii=False,
            ) + "\n"
            await self.tools.execute("append_file", {"path": self.config.auto_log_file, "content": log_line})

        self.context.record_decision_summary(
            f"{cv_event.event_type}@{cv_event.zone}:{decision_group.risk_level.value}->{decision_group.decisions[0].action}"
        )

        lat_ms = (time.perf_counter() - t0) * 1000.0
        self._latencies_ms.append(lat_ms)
        self._stats["events_processed"] += 1

        result = CVAgentResult(
            event=cv_event,
            context=snapshot,
            decision_group=decision_group,
            tool_results=tool_results,
            alert_dispatched=alert_dispatched,
            used_moe=used_moe,
            latency_ms=lat_ms,
        )

        # 8. Fire user event hooks
        for cb in list(self._hooks.get(cv_event.event_type, [])) + list(self._hooks.get("*", [])):
            out = cb(result)
            if inspect.isawaitable(out):
                await out

        return result

    def handle_event_sync(
        self,
        event: Union[CVEvent, Dict[str, Any]],
        *,
        force_moe: bool = False,
        extra_context: Optional[Dict[str, Any]] = None,
    ) -> CVAgentResult:
        """Synchronous entrypoint for classic OpenCV `while cap.isOpened():` frame loops."""
        coro = self.handle_event(event, force_moe=force_moe, extra_context=extra_context)
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        raise RuntimeError(
            "handle_event_sync() cannot be called inside an active event loop; use `await handle_event(...)`."
        )

    async def process_stream(
        self,
        events: AsyncIterable[Union[CVEvent, Dict[str, Any]]],
        *,
        max_events: Optional[int] = None,
    ) -> List[CVAgentResult]:
        """Process a live async CV event stream continuously with bounded O(1) memory."""
        results: List[CVAgentResult] = []
        count = 0
        async for ev in events:
            res = await self.handle_event(ev)
            results.append(res)
            count += 1
            if max_events is not None and count >= max_events:
                break
        return results

    def get_stats(self) -> Dict[str, Any]:
        lats = sorted(self._latencies_ms)
        p50 = lats[len(lats) // 2] if lats else 0.0
        p99 = lats[min(len(lats) - 1, int(len(lats) * 0.99))] if lats else 0.0
        return {
            **self._stats,
            "latency_p50_ms": round(p50, 3),
            "latency_p99_ms": round(p99, 3),
            "context": self.context.stats(),
            "connected_databases": self.db.connected_aliases(),
            "enabled_tools": self.tools.enabled_names(),
            "model": self.llm.info() if self.llm is not None else None,
        }

    def close(self) -> None:
        """Cleanly close all local database handles."""
        self.db.close_all()


MicroCVAgent = CVAgent
