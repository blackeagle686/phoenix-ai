# Phoenix Micron CV Agent (`phoenix.framework.cv_agent`)

**Micron CV Agent** (`CVAgent` / `MicroCVAgent`) is an ultra-lightweight, event-driven edge AI agent built to run **side-by-side with Computer Vision pipelines** (YOLO, OpenCV, ONNX Runtime, TFLite, MediaPipe, DeepStream, Frigate) on **mobile phone CPUs, Raspberry Pi, and low-RAM edge devices**.

---

## Core Design Guarantees

| Guarantee | Implementation |
|---|---|
| **Zero Heavy Startup Imports** | Core engine uses **Python stdlib only** (`dataclasses`, `collections.deque`, `sqlite3`, `json`, `urllib.request`). Does **not** import `torch`, `transformers`, `numpy`, `cv2`, `sklearn`, or `pydantic`. |
| **Strict `<= 1B` MoE Model Policy** | Enforced by `validate_micro_moe_model()` (`HARD_MAX_MOE_PARAMS_B = 1.0`). Rejects dense models and rejects any MoE model `> 1.0B` with `MicroMoEPolicyError` **before** allocating RAM. |
| **Fixed O(1) Memory (24/7 Safe)** | `SlidingCVContext` uses bounded `deque` ring buffers and bounded track/alert tables so RAM stays flat even after millions of camera frames. |
| **Tiny Decisions -> Decision Groups** | Combines sub-millisecond deterministic `ContextDecisionRule`s with `<= 1B` MoE micro-inference to fill a complete, validated `DecisionGroup` (atomic `TinyDecision`s + contextual slots + tool calls). |
| **User-Modifiable CV Tools** | Add, replace, customize defaults, enable/disable, or delete any built-in or custom CV tool at runtime (`@agent.tool`, `register_tool`, `modify_tool`, `remove_tool`). |
| **Local Storage & Databases** | Built-in symlink-jailed local file read/write (`LocalFileStore`), multi-channel `AlertDispatcher` with cooldown anti-spam, and zero-dependency local SQLite (`LocalSQLiteStore`). |

---

## Architecture Pipeline

```text
  CV Detector (YOLO / OpenCV / ONNX / TFLite / RTSP Stream)
                          │
                          ▼  CVEvent (from_yolo / from_boxes / from_dict)
              ┌───────────────────────┐
              │   SlidingCVContext    │  Bounded deque ring buffer
              │  bursts / track dwell │  + alert cooldown tracker
              └───────────┬───────────┘
                          │
                          ▼  Optional local SQLite lookup (`set_db_enricher`)
              ┌───────────────────────┐
              │ ContextDecisionRules  │  < 0.1ms zero-LLM fast path
              └───────────┬───────────┘
                          │ (if no rule bypassed)
                          ▼
              ┌───────────────────────┐
              │   MicroMoELLM (<=1B)  │  Strict MoE <= 1B policy
              │   GGUF CPU / Callable │  n_ctx <= 2048, compact JSON or KV
              └───────────┬───────────┘
                          │
                          ▼  Validated DecisionGroup (TinyDecisions + Context Slots)
        ┌─────────────────┼──────────────────┬──────────────────┐
        ▼                 ▼                  ▼                  ▼
   send_alert      read/write files    local SQLite DB     Custom CV Tools
  (with cooldown)  (sandboxed root)   (read/query/log)   (gate/siren/MQTT)
```

---

## 60-Second Quickstart

### In a Script / OpenCV Loop (Sync)
```python
from phoenix.framework.cv_agent import CVAgent, CVAgentConfig, CVEvent, Detection

agent = CVAgent(
    model="ibm-granite/granite-3.1-1b-a400m-instruct",  # or alias "granite-1b-moe"
    config=CVAgentConfig(
        workspace_root="./cv_edge_workspace",
        alert_cooldown_seconds=20.0,
        auto_log_file="logs/cv_decisions.jsonl",
    ),
    db_path="./cv_edge_workspace/edge_security.db",
    on_alert=lambda alert: print(f"[ALERT][{alert['severity']}] {alert['message']}"),
)

event = CVEvent(
    event_type="person_detected",
    camera_id="cam_entrance",
    zone="restricted_gate",
    detections=[Detection(label="person", confidence=0.94, track_id="p_12")],
)

result = agent.handle_event_sync(event)
print(result.decision_group.to_dict())
agent.close()
```

### In Jupyter / Kaggle / Async Services
```python
result = await agent.handle_event(event)
print(result.decision_group.to_dict())
print(agent.get_stats())
```

> Both `await agent.handle_event(event)` and `agent.handle_event_sync(event)` work in Jupyter/Kaggle/Colab notebooks as well as standard Python scripts.

---

## Documentation Index

1. **[Supported Models & `<= 1B` MoE Policy](01_supported_models.md)** — Exact allowed model IDs, aliases, RAM footprints, phone CPU settings, and rejected model rules.
2. **[Tools, Alerts, Local Files & SQLite Databases](02_tools_database_and_files.md)** — How to modify/replace tools, configure alert channels & cooldowns, read/write local files safely, and connect/query local SQLite records.
3. **[Real-World CV Scenarios & Production Recipes](03_real_world_scenarios.md)** — End-to-end code for Smart Gate ALPR, Factory PPE + YOLO, Mobile/Pi Perimeter Loitering + OpenCV, Async RTSP Streams, and Zero-RAM Battery Mode.
