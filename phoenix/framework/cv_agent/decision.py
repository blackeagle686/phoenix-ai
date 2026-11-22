"""Tiny Decisions, Context Decision Groups, Rule Engine & <=1B MoE Micro-Parser.

A <=1B MoE model cannot run heavy multi-turn ReAct loops on a phone CPU, so CVAgent
decomposes every vision event into:
  1. Fast deterministic `ContextDecisionRule` checks (< 0.1ms CPU, 0 MoE tokens)
  2. Ultra-compact <=1B MoE micro-inference filling a structured `DecisionGroup`
     (a coherent set of tiny decisions + contextual fields + concrete micro tool calls)
  3. Dual-format resilient parsing (flat JSON OR `KEY=VALUE` lines) + automatic
     context slot filling so weak/tiny models NEVER break downstream automation.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from phoenix.framework.cv_agent.context import ContextSnapshot
from phoenix.framework.cv_agent.events import CVEvent, CVSeverity

_DC_SLOTS: Dict[str, Any] = {"slots": True} if sys.version_info >= (3, 10) else {}


@dataclass(**_DC_SLOTS)
class MicroToolCall:
    """Single lightweight tool invocation inside a DecisionGroup."""

    tool_name: str
    arguments: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"tool_name": self.tool_name, "arguments": dict(self.arguments)}


@dataclass(**_DC_SLOTS)
class TinyDecision:
    """Atomic micro-decision targeting a specific CV context dimension."""

    action: str  # ignore | log | alert | query_db | write_file | escalate | custom
    severity: CVSeverity = CVSeverity.LOW
    confidence: float = 0.8
    reason: str = ""
    target: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "action": self.action,
            "severity": self.severity.value,
            "confidence": round(float(self.confidence), 3),
            "reason": self.reason,
        }
        if self.target:
            out["target"] = self.target
        return out


@dataclass(**_DC_SLOTS)
class DecisionGroup:
    """Complete filled group of contextual micro-decisions produced for a CVEvent."""

    event_id: str
    camera_id: str
    zone: str
    risk_level: CVSeverity = CVSeverity.LOW
    summary: str = ""
    alert_needed: bool = False
    alert_message: str = ""
    decisions: List[TinyDecision] = field(default_factory=list)
    context_slots: Dict[str, Any] = field(default_factory=dict)
    tool_calls: List[MicroToolCall] = field(default_factory=list)
    source: str = "micro_moe"  # rule_fast_path | micro_moe | hybrid_filled

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "camera_id": self.camera_id,
            "zone": self.zone,
            "risk_level": self.risk_level.value,
            "summary": self.summary,
            "alert_needed": self.alert_needed,
            "alert_message": self.alert_message,
            "decisions": [d.to_dict() for d in self.decisions],
            "context_slots": dict(self.context_slots),
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "source": self.source,
        }


@dataclass(**_DC_SLOTS)
class ContextDecisionRule:
    """User-customizable deterministic micro-rule evaluated before waking the <=1B MoE."""

    name: str
    predicate: Callable[[CVEvent, ContextSnapshot], bool]
    builder: Callable[[CVEvent, ContextSnapshot], DecisionGroup]
    priority: int = 50
    bypass_moe: bool = True  # If True, resolves event at 0ms without calling the MoE


def _coerce_severity(raw: Any, default: CVSeverity = CVSeverity.LOW) -> CVSeverity:
    if isinstance(raw, CVSeverity):
        return raw
    val = str(raw or "").strip().lower()
    mapping = {
        "info": CVSeverity.INFO,
        "none": CVSeverity.INFO,
        "low": CVSeverity.LOW,
        "med": CVSeverity.MEDIUM,
        "medium": CVSeverity.MEDIUM,
        "warn": CVSeverity.MEDIUM,
        "warning": CVSeverity.MEDIUM,
        "high": CVSeverity.HIGH,
        "critical": CVSeverity.CRITICAL,
        "crit": CVSeverity.CRITICAL,
        "emergency": CVSeverity.CRITICAL,
    }
    return mapping.get(val, default)


def _extract_flat_json(text: str) -> Optional[Dict[str, Any]]:
    raw = (text or "").strip()
    if not raw:
        return None
    # Strip optional code fences or <think> tags
    if "<think>" in raw:
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    fence = re.search(r"```(?:json)?\s*\r?\n(.*?)```", raw, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        raw = fence.group(1).strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    cand = raw[start : end + 1]
    try:
        obj = json.loads(cand)
        return obj if isinstance(obj, dict) else None
    except Exception:
        # Attempt minor cleanup (trailing commas / single quotes)
        cleaned = re.sub(r",\s*([}\]])", r"\1", cand)
        try:
            obj = json.loads(cleaned)
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None


def _parse_kv_fallback(text: str) -> Dict[str, Any]:
    """Parse ultra-simple `KEY: VALUE` / `KEY=VALUE` outputs emitted by <=1B MoE models."""
    out: Dict[str, Any] = {}
    for line in (text or "").splitlines():
        m = re.match(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*[:=]\s*(.+?)\s*$", line)
        if not m:
            continue
        k = m.group(1).strip().lower()
        v = m.group(2).strip().strip("'\"")
        if v.lower() in ("true", "yes", "1"):
            out[k] = True
        elif v.lower() in ("false", "no", "0"):
            out[k] = False
        else:
            out[k] = v
    return out


class MicroDecisionEngine:
    """Prompt builder, dual-format parser, and context-group filler for <= 1B MoE models."""

    @staticmethod
    def build_micro_prompt(
        event: CVEvent,
        snapshot: ContextSnapshot,
        *,
        available_tools: List[str],
        context_schema_keys: Optional[List[str]] = None,
        policy_hint: str = "",
    ) -> str:
        """Build a tiny (<240 tokens) structured prompt engineered specifically for <=1B MoE models."""
        tools_str = ", ".join(available_tools[:12]) if available_tools else "send_alert, write_file, read_file, db_read_records"
        slot_keys = context_schema_keys or ["subject", "zone_status", "recommended_next"]
        lines = [
            "CV MICRO-DECISION TASK: Analyze event + context and fill a compact JSON decision group.",
            f"EVENT: {event.to_micro_line()}",
            snapshot.to_micro_prompt_block(),
            f"TOOLS: {tools_str}",
        ]
        if policy_hint:
            lines.append(f"POLICY: {policy_hint[:240]}")
        lines.append(
            "REPLY ONLY VALID COMPACT JSON:\n"
            '{"risk":"low|medium|high|critical","summary":"short reason","alert":true|false,'
            '"alert_msg":"short alert if needed",'
            '"decisions":[{"action":"log|alert|ignore|escalate|query_db|write_file","severity":"low|medium|high|critical","reason":"short"}],'
            f'"context_slots":{{{", ".join(f"{json.dumps(k)}:\"value\"" for k in slot_keys[:5])}}},'
            '"tools":[{"tool":"tool_name","args":{}}]}'
        )
        return "\n".join(lines)

    @staticmethod
    def parse_and_fill_decision_group(
        raw_reply: str,
        event: CVEvent,
        snapshot: ContextSnapshot,
        *,
        allowed_tools: Optional[set] = None,
        required_context_slots: Optional[List[str]] = None,
        source: str = "micro_moe",
    ) -> DecisionGroup:
        """Parse <=1B MoE output (JSON or KV fallback) and guarantee every context slot is cleanly filled."""
        data = _extract_flat_json(raw_reply)
        if data is None:
            data = _parse_kv_fallback(raw_reply)

        risk = _coerce_severity(
            data.get("risk") or data.get("risk_level") or data.get("severity"),
            default=event.severity_hint,
        )
        summary = str(data.get("summary") or data.get("reason") or f"{event.event_type} at {event.camera_id}/{event.zone}").strip()[:200]

        raw_alert = data.get("alert", data.get("alert_needed"))
        if isinstance(raw_alert, bool):
            alert_needed = raw_alert
        elif isinstance(raw_alert, str):
            alert_needed = raw_alert.strip().lower() in ("true", "yes", "1", "alert")
        else:
            alert_needed = risk in (CVSeverity.HIGH, CVSeverity.CRITICAL)

        # Respect alert cooldown unless CRITICAL
        if snapshot.alert_on_cooldown and risk != CVSeverity.CRITICAL:
            alert_needed = False

        alert_msg = str(data.get("alert_msg") or data.get("alert_message") or "").strip()
        if alert_needed and not alert_msg:
            alert_msg = f"[{risk.value.upper()}] {event.event_type} on {event.camera_id} ({event.zone}): {summary}"

        # Parse atomic TinyDecisions
        decisions: List[TinyDecision] = []
        raw_decisions = data.get("decisions")
        if isinstance(raw_decisions, list):
            for item in raw_decisions[:6]:
                if isinstance(item, dict):
                    decisions.append(
                        TinyDecision(
                            action=str(item.get("action") or "log").strip().lower(),
                            severity=_coerce_severity(item.get("severity"), default=risk),
                            confidence=float(item.get("confidence", event.max_confidence or 0.75)),
                            reason=str(item.get("reason") or summary)[:160],
                            target=str(item["target"]) if item.get("target") else None,
                        )
                    )
                elif isinstance(item, str) and item.strip():
                    decisions.append(TinyDecision(action=item.strip().lower(), severity=risk, reason=summary))
        elif isinstance(data.get("action"), str) and data["action"].strip():
            decisions.append(
                TinyDecision(
                    action=data["action"].strip().lower(),
                    severity=risk,
                    confidence=event.max_confidence or 0.75,
                    reason=summary,
                )
            )

        if not decisions:
            default_act = "alert" if alert_needed else "log"
            decisions.append(
                TinyDecision(
                    action=default_act,
                    severity=risk,
                    confidence=event.max_confidence or 0.75,
                    reason=summary,
                )
            )

        # Parse & whitelist tool calls
        tool_calls: List[MicroToolCall] = []
        raw_tools = data.get("tools") or data.get("tool_calls") or []
        if isinstance(raw_tools, list):
            for tc in raw_tools[:6]:
                if not isinstance(tc, dict):
                    continue
                tname = str(tc.get("tool") or tc.get("tool_name") or tc.get("name") or "").strip()
                if not tname:
                    continue
                if allowed_tools is not None and tname not in allowed_tools:
                    continue
                args = tc.get("args") or tc.get("arguments") or {}
                tool_calls.append(MicroToolCall(tool_name=tname, arguments=dict(args) if isinstance(args, dict) else {}))

        # Fill contextual decision slots deterministically so group is always complete
        raw_slots = data.get("context_slots") if isinstance(data.get("context_slots"), dict) else {}
        filled_slots: Dict[str, Any] = dict(raw_slots)
        auto_defaults: Dict[str, Any] = {
            "camera_id": event.camera_id,
            "zone": event.zone,
            "event_type": event.event_type,
            "labels": list(event.labels),
            "burst_count": snapshot.same_type_in_window,
            "max_dwell_sec": round(snapshot.max_track_dwell_sec, 2),
            "alert_on_cooldown": snapshot.alert_on_cooldown,
            "db_matched_count": len(snapshot.db_facts),
            "subject": ", ".join(event.labels[:3]) if event.labels else event.event_type,
            "zone_status": "alert" if alert_needed else ("monitored" if snapshot.same_zone_in_window > 1 else "normal"),
            "recommended_next": decisions[0].action,
        }
        for k, v in auto_defaults.items():
            if k not in filled_slots or filled_slots[k] in ("", None):
                filled_slots[k] = v
        for req_k in (required_context_slots or []):
            if req_k not in filled_slots or filled_slots[req_k] in ("", None):
                filled_slots[req_k] = auto_defaults.get(req_k, "unknown")

        return DecisionGroup(
            event_id=event.event_id,
            camera_id=event.camera_id,
            zone=event.zone,
            risk_level=risk,
            summary=summary,
            alert_needed=alert_needed,
            alert_message=alert_msg,
            decisions=decisions,
            context_slots=filled_slots,
            tool_calls=tool_calls,
            source=source,
        )
