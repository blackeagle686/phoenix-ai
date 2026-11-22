"""Fixed-memory Sliding Temporal Context Window for real-time Computer Vision streams.

Designed for 24/7 operation on low-RAM phone/edge CPUs:
  - Bounded `deque` ring buffers guarantee O(1) memory growth forever.
  - Computes temporal burst counts, track dwell durations, and alert cooldowns
    deterministically in sub-millisecond Python before any <=1B MoE call.
"""

from __future__ import annotations

import sys
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

from phoenix.framework.cv_agent.events import CVEvent

_DC_SLOTS: Dict[str, Any] = {"slots": True} if sys.version_info >= (3, 10) else {}


@dataclass(**_DC_SLOTS)
class ContextSnapshot:
    """Compact contextual summary attached to a single CVEvent for decision filling."""

    camera_id: str
    zone: str
    event_type: str
    recent_event_count: int
    same_type_in_window: int
    same_zone_in_window: int
    max_track_dwell_sec: float
    active_tracks: Dict[str, float] = field(default_factory=dict)
    label_counts_window: Dict[str, int] = field(default_factory=dict)
    alert_on_cooldown: bool = False
    recent_history_lines: List[str] = field(default_factory=list)
    recent_decisions: List[str] = field(default_factory=list)
    db_facts: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "zone": self.zone,
            "event_type": self.event_type,
            "recent_event_count": self.recent_event_count,
            "same_type_in_window": self.same_type_in_window,
            "same_zone_in_window": self.same_zone_in_window,
            "max_track_dwell_sec": round(self.max_track_dwell_sec, 2),
            "active_tracks": {k: round(v, 2) for k, v in self.active_tracks.items()},
            "label_counts_window": dict(self.label_counts_window),
            "alert_on_cooldown": self.alert_on_cooldown,
            "db_facts": list(self.db_facts[:6]),
        }

    def to_micro_prompt_block(self) -> str:
        """Render a minimal (<120 tokens) context block suitable for <=1B MoE models."""
        parts = [
            f"CTX: cam={self.camera_id} zone={self.zone} burst_same_type={self.same_type_in_window} "
            f"zone_events={self.same_zone_in_window} max_dwell_s={self.max_track_dwell_sec:.1f} "
            f"alert_cooldown={self.alert_on_cooldown}"
        ]
        if self.label_counts_window:
            top_labels = sorted(self.label_counts_window.items(), key=lambda x: -x[1])[:6]
            parts.append("WINDOW_OBJS: " + ", ".join(f"{k}:{v}" for k, v in top_labels))
        if self.db_facts:
            fact_lines = []
            for row in self.db_facts[:4]:
                compact = ", ".join(f"{k}={v}" for k, v in list(row.items())[:6])
                fact_lines.append("  - " + compact)
            parts.append("DB_RECORDS:\n" + "\n".join(fact_lines))
        if self.recent_history_lines:
            parts.append("RECENT_EVENTS:\n" + "\n".join(f"  - {ln}" for ln in self.recent_history_lines[-4:]))
        if self.recent_decisions:
            parts.append("RECENT_DECISIONS: " + " | ".join(self.recent_decisions[-3:]))
        return "\n".join(parts)


class SlidingCVContext:
    """Fixed-RAM temporal ring buffer tracking CV events, track dwell times, and alert cooldowns."""

    def __init__(
        self,
        *,
        max_events: int = 32,
        window_seconds: float = 60.0,
        max_tracks: int = 64,
        max_decisions: int = 16,
        alert_cooldown_seconds: float = 15.0,
    ):
        self.max_events = max(4, int(max_events))
        self.window_seconds = max(1.0, float(window_seconds))
        self.max_tracks = max(8, int(max_tracks))
        self.max_decisions = max(4, int(max_decisions))
        self.alert_cooldown_seconds = max(0.0, float(alert_cooldown_seconds))

        self._events: Deque[CVEvent] = deque(maxlen=self.max_events)
        self._decisions: Deque[str] = deque(maxlen=self.max_decisions)
        # track_key -> (first_seen_ts, last_seen_ts)
        self._tracks: Dict[str, Tuple[float, float]] = {}
        # cooldown_key -> last_alert_ts
        self._alert_timestamps: Dict[str, float] = {}

    def record_event(self, event: CVEvent) -> ContextSnapshot:
        """Ingest a CVEvent into the bounded ring buffer and compute its temporal context snapshot."""
        now = float(event.timestamp or time.time())
        self._events.append(event)
        self._update_tracks(event, now)
        return self.build_snapshot(event, now=now)

    def _update_tracks(self, event: CVEvent, now: float) -> None:
        # Prune stale tracks outside 2x window so dict stays strictly bounded
        cutoff = now - (self.window_seconds * 2.0)
        stale = [k for k, (_, last_ts) in self._tracks.items() if last_ts < cutoff]
        for k in stale:
            self._tracks.pop(k, None)

        for det in event.detections:
            if det.track_id is None:
                continue
            tkey = f"{event.camera_id}:{det.zone or event.zone}:{det.track_id}"
            if tkey in self._tracks:
                first_ts, _ = self._tracks[tkey]
                self._tracks[tkey] = (first_ts, now)
            else:
                if len(self._tracks) >= self.max_tracks:
                    oldest_k = min(self._tracks.items(), key=lambda kv: kv[1][1])[0]
                    self._tracks.pop(oldest_k, None)
                self._tracks[tkey] = (now, now)

    def build_snapshot(
        self,
        event: CVEvent,
        *,
        now: Optional[float] = None,
        db_facts: Optional[List[Dict[str, Any]]] = None,
    ) -> ContextSnapshot:
        ts_now = float(now if now is not None else (event.timestamp or time.time()))
        cutoff = ts_now - self.window_seconds

        recent_in_window = [e for e in self._events if e.timestamp >= cutoff]
        same_type = sum(
            1 for e in recent_in_window
            if e.event_type == event.event_type and e.camera_id == event.camera_id
        )
        same_zone = sum(
            1 for e in recent_in_window
            if e.zone == event.zone and e.camera_id == event.camera_id
        )

        label_counts: Dict[str, int] = {}
        for e in recent_in_window:
            if e.camera_id != event.camera_id:
                continue
            for det in e.detections:
                label_counts[det.label] = label_counts.get(det.label, 0) + 1

        active_tracks: Dict[str, float] = {}
        max_dwell = 0.0
        for det in event.detections:
            if det.track_id is None:
                continue
            tkey = f"{event.camera_id}:{det.zone or event.zone}:{det.track_id}"
            pair = self._tracks.get(tkey)
            if pair is not None:
                dwell = max(0.0, ts_now - pair[0])
                active_tracks[det.track_id] = dwell
                if dwell > max_dwell:
                    max_dwell = dwell

        cooldown_key = f"{event.camera_id}:{event.zone}:{event.event_type}"
        last_alert = self._alert_timestamps.get(cooldown_key)
        on_cooldown = (
            last_alert is not None
            and self.alert_cooldown_seconds > 0.0
            and (ts_now - last_alert) < self.alert_cooldown_seconds
        )

        history_lines = [e.to_micro_line() for e in recent_in_window[-5:-1]]
        return ContextSnapshot(
            camera_id=event.camera_id,
            zone=event.zone,
            event_type=event.event_type,
            recent_event_count=len(recent_in_window),
            same_type_in_window=same_type,
            same_zone_in_window=same_zone,
            max_track_dwell_sec=max_dwell,
            active_tracks=active_tracks,
            label_counts_window=label_counts,
            alert_on_cooldown=on_cooldown,
            recent_history_lines=history_lines,
            recent_decisions=list(self._decisions),
            db_facts=list(db_facts or []),
        )

    def mark_alert_sent(self, event: CVEvent, ts: Optional[float] = None) -> None:
        now = float(ts if ts is not None else time.time())
        key = f"{event.camera_id}:{event.zone}:{event.event_type}"
        self._alert_timestamps[key] = now
        # Keep alert map bounded
        if len(self._alert_timestamps) > 128:
            oldest = min(self._alert_timestamps.items(), key=lambda kv: kv[1])[0]
            self._alert_timestamps.pop(oldest, None)

    def record_decision_summary(self, summary: str) -> None:
        if summary:
            self._decisions.append(summary[:140])

    def clear(self) -> None:
        self._events.clear()
        self._decisions.clear()
        self._tracks.clear()
        self._alert_timestamps.clear()

    def stats(self) -> Dict[str, Any]:
        return {
            "buffered_events": len(self._events),
            "max_events": self.max_events,
            "tracked_objects": len(self._tracks),
            "decision_history": len(self._decisions),
            "window_seconds": self.window_seconds,
        }
