# Real-World Computer Vision Scenarios & Recipes

These complete, copy-pasteable scenarios demonstrate how `CVAgent` runs side-by-side with real CV detectors on phone CPUs and low-RAM edge devices.

---

## Scenario 1: Smart Gate ALPR (Plate Recognition + Local SQLite + Gate Lock + Alert Cooldown)

**Situation**: An edge camera runs a plate detector (YOLO + OCR) at a facility entrance. The Micron CV Agent checks each detected plate against a local SQLite `plates` table, fills a `DecisionGroup`, locks the barrier if the plate is blocked, writes an incident file, and sends a deduplicated alert.

```python
import asyncio
import sqlite3
from phoenix.framework.cv_agent import CVAgent, CVAgentConfig, CVEvent, Detection

# 1. Prepare a local SQLite database
conn = sqlite3.connect("./gate_edge/security.db")
conn.execute("CREATE TABLE IF NOT EXISTS plates (plate TEXT PRIMARY KEY, owner TEXT, status TEXT)")
conn.execute("INSERT OR REPLACE INTO plates VALUES ('ABC-123', 'Sara', 'authorized')")
conn.execute("INSERT OR REPLACE INTO plates VALUES ('BLK-909', 'Unknown', 'blocked')")
conn.commit()
conn.close()

async def main():
    agent = CVAgent(
        model="ibm-granite/granite-3.1-1b-a400m-instruct",  # <= 1B MoE (400M active)
        config=CVAgentConfig(
            workspace_root="./gate_edge",
            alert_cooldown_seconds=30.0,
            auto_log_file="logs/gate_decisions.jsonl",
            policy_hint="Block and alert immediately if DB plate status=blocked. Log silently if status=authorized.",
            context_slots=["subject", "plate_status", "zone_status", "recommended_next"],
        ),
        db_path="./gate_edge/security.db",
        on_alert=lambda a: print(f"🚨 [{a['severity'].upper()}] {a['message']}"),
        alert_log_path="./gate_edge/alerts.jsonl",
    )

    # Custom CV hardware tool for the gate controller
    @agent.tool(name="control_barrier", description="Open or lock gate barrier (gate_id, action='lock'|'open').")
    def control_barrier(gate_id: str, action: str = "lock"):
        print(f"🚧 Barrier [{gate_id}] -> {action.upper()}")
        return {"gate_id": gate_id, "state": action}

    # Auto-enrich every plate event from local SQLite before the <= 1B MoE decides
    agent.set_db_enricher(
        lambda ev, db: db.read_records("plates", where={"plate": ev.metadata["plate"]}, limit=2)["records"]
        if ev.metadata.get("plate") else []
    )

    # CV detector emits a plate detection event
    event = CVEvent(
        event_type="vehicle_plate_detected",
        camera_id="cam_north_gate",
        zone="entry_lane_1",
        detections=[Detection(label="car", confidence=0.95, track_id="veh_17")],
        metadata={"plate": "BLK-909"},
    )

    result = await agent.handle_event(event)
    print("Decision Group:", result.decision_group.to_dict())
    print("Tool Results:", [r.to_dict() for r in result.tool_results])
    agent.close()

asyncio.run(main())
```

---

## Scenario 2: Factory PPE Safety Monitoring with Ultralytics YOLO (`CVEvent.from_yolo`)

**Situation**: A factory floor camera runs YOLOv8n/YOLO11n detecting workers, helmets, and vests. Critical high-confidence fire/smoke events bypass the MoE via a `< 0.1ms` `ContextDecisionRule`, while PPE violations wake the `<= 1B` MoE to inspect shift records in SQLite and dispatch targeted alerts.

```python
from phoenix.framework.cv_agent import (
    CVAgent,
    CVAgentConfig,
    CVEvent,
    CVSeverity,
    ContextDecisionRule,
    DecisionGroup,
    MicroToolCall,
    TinyDecision,
)

# Instant 0ms safety rule (bypasses <= 1B MoE for immediate life-safety events)
fire_rule = ContextDecisionRule(
    name="instant_fire_override",
    priority=1,
    bypass_moe=True,
    predicate=lambda ev, ctx: any(d.label in ("fire", "smoke") and d.confidence >= 0.70 for d in ev.detections),
    builder=lambda ev, ctx: DecisionGroup(
        event_id=ev.event_id,
        camera_id=ev.camera_id,
        zone=ev.zone,
        risk_level=CVSeverity.CRITICAL,
        summary="Immediate fire/smoke detected on factory floor",
        alert_needed=True,
        alert_message=f"CRITICAL FIRE/SMOKE on {ev.camera_id} ({ev.zone})!",
        decisions=[
            TinyDecision(action="alert", severity=CVSeverity.CRITICAL, reason="Fire confidence >= 0.70"),
            TinyDecision(action="write_file", severity=CVSeverity.CRITICAL, reason="Log emergency timestamp"),
        ],
        context_slots={"zone_status": "evacuate", "recommended_next": "sound_fire_alarm"},
        tool_calls=[
            MicroToolCall("append_file", {"path": "emergencies.log", "content": f"FIRE {ev.camera_id} {ev.zone}\n"}),
        ],
        source="rule_fast_path",
    ),
)

agent = CVAgent(
    model="granite-1b-moe",
    config=CVAgentConfig(workspace_root="./factory_safety", alert_cooldown_seconds=20.0),
    rules=[fire_rule],
    on_alert=lambda a: print("FACTORY ALERT:", a["severity"], a["message"]),
)

# Inside your Ultralytics YOLO loop:
# from ultralytics import YOLO
# yolo = YOLO("ppe_nano.pt")
# for frame_idx, r in enumerate(yolo.track(source=0, stream=True)):
#     ev = CVEvent.from_yolo(
#         r,
#         event_type="ppe_frame_check",
#         camera_id="floor_cam_2",
#         zone="welding_bay",
#         frame_id=frame_idx,
#         min_confidence=0.40,
#     )
#     if ev.detections:
#         res = agent.handle_event_sync(ev)
#         if res.alert_dispatched:
#             print("Alert sent:", res.decision_group.summary)
```

---

## Scenario 3: Mobile Phone / Raspberry Pi Perimeter Loitering (OpenCV + Dwell Time)

**Situation**: An Android phone (Termux) or Raspberry Pi runs a lightweight OpenCV / TFLite person detector. `SlidingCVContext` tracks how long each `track_id` stays in a restricted zone (`max_track_dwell_sec`). If someone loiters for more than 10 seconds, the agent logs the event and sends a phone notification.

```python
import time
from phoenix.framework.cv_agent import (
    CVAgent,
    CVAgentConfig,
    CVEvent,
    CVSeverity,
    ContextDecisionRule,
    DecisionGroup,
    TinyDecision,
)

# Rule triggers only when a tracked person lingers >= 10 seconds in the restricted zone
loitering_rule = ContextDecisionRule(
    name="zone_loitering_alert",
    priority=10,
    bypass_moe=True,
    predicate=lambda ev, ctx: ev.zone == "back_door" and ctx.max_track_dwell_sec >= 10.0,
    builder=lambda ev, ctx: DecisionGroup(
        event_id=ev.event_id,
        camera_id=ev.camera_id,
        zone=ev.zone,
        risk_level=CVSeverity.HIGH,
        summary=f"Person loitering at {ev.zone} for {ctx.max_track_dwell_sec:.1f}s",
        alert_needed=not ctx.alert_on_cooldown,
        alert_message=f"Loitering detected at {ev.zone} ({ctx.max_track_dwell_sec:.1f}s)!",
        decisions=[
            TinyDecision(action="alert", severity=CVSeverity.HIGH, reason="Dwell threshold exceeded"),
            TinyDecision(action="log", severity=CVSeverity.MEDIUM, reason="Record dwell duration"),
        ],
        context_slots={"zone_status": "loitering", "dwell_sec": round(ctx.max_track_dwell_sec, 1)},
        source="rule_fast_path",
    ),
)

agent = CVAgent(
    model="granite-1b-moe",
    config=CVAgentConfig(
        workspace_root="./phone_guard",
        max_context_events=24,        # Fixed tiny ring buffer for phone RAM
        context_window_seconds=60.0,
        alert_cooldown_seconds=30.0,
    ),
    rules=[loitering_rule],
    on_alert=lambda a: print("📱 PHONE PUSH:", a["message"]),
)

# Inside an OpenCV / TFLite frame loop using raw bounding boxes:
base_ts = time.time()
for step in range(4):
    ev = CVEvent.from_boxes(
        boxes=[[120, 80, 260, 410]],
        labels=["person"],
        scores=[0.91],
        track_ids=["trk_5"],
        event_type="person_present",
        camera_id="phone_rear_cam",
        zone="back_door",
    )
    ev.timestamp = base_ts + step * 4.0  # 0s, 4s, 8s, 12s
    res = agent.handle_event_sync(ev)
    print(f"t+{step*4}s | dwell={res.context.max_track_dwell_sec:.1f}s | alert={res.alert_dispatched}")

agent.close()
```

---

## Scenario 4: Continuous Async Multi-Camera Stream (`process_stream`)

**Situation**: An edge gateway ingests JSON events from multiple RTSP cameras (e.g., Frigate / DeepStream / Redis / MQTT) asynchronously, stores incident rows in local SQLite, and forwards alerts via a customized webhook tool.

```python
import asyncio
from phoenix.framework.cv_agent import CVAgent, CVAgentConfig

async def camera_event_feed():
    sample_events = [
        {
            "event_type": "line_crossing",
            "camera_id": "cam_dock_1",
            "zone": "loading_bay",
            "severity_hint": "medium",
            "detections": [{"label": "forklift", "confidence": 0.93, "track_id": "f_3"}],
        },
        {
            "event_type": "after_hours_entry",
            "camera_id": "cam_vault",
            "zone": "server_room",
            "severity_hint": "high",
            "detections": [{"label": "person", "confidence": 0.96, "track_id": "p_9"}],
        },
    ]
    for ev in sample_events:
        yield ev
        await asyncio.sleep(0.05)

async def main():
    agent = CVAgent(
        model="granite-1b-moe",
        config=CVAgentConfig(workspace_root="./gateway_edge", auto_log_file="stream_audit.jsonl"),
        db_path="./gateway_edge/events.db",
    )

    # Subscribe a post-decision hook for every event
    agent.on_event("*", lambda r: print(f"[{r.event.camera_id}] risk={r.decision_group.risk_level.value} ({r.latency_ms:.1f}ms)"))

    results = await agent.process_stream(camera_event_feed())
    print("Processed:", len(results), "| Stats:", agent.get_stats())
    agent.close()

asyncio.run(main())
```

---

## Scenario 5: Zero-RAM Ultra-Edge Battery Mode (`model=None`)

**Situation**: Running on a 512 MB RAM IoT board or low-end phone where even a 300M GGUF model is unnecessary because your local SQLite records + deterministic rules cover 100% of decisions.

```python
from phoenix.framework.cv_agent import CVAgent, CVAgentConfig, CVEvent, CVSeverity, Detection

agent = CVAgent(
    model=None,  # Zero LLM weights loaded (< 2 MB Python footprint!)
    config=CVAgentConfig(workspace_root="./tiny_iot", auto_log_file="events.jsonl"),
    db_path="./tiny_iot/sensors.db",
    on_alert=lambda a: print("LOW-POWER ALERT:", a["message"]),
)

res = agent.handle_event_sync(
    CVEvent(
        event_type="intrusion",
        camera_id="cam_shed",
        zone="perimeter",
        severity_hint=CVSeverity.HIGH,
        detections=[Detection(label="person", confidence=0.89, track_id="p_1")],
    )
)
print("Risk:", res.decision_group.risk_level.value, "| Alert sent:", res.alert_dispatched)
agent.close()
```
