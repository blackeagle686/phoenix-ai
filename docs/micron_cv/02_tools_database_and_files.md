# Tools, Alerts, Local Files & Local Databases

Every built-in capability in `CVAgent` is exposed through a mutable **`MicroToolRegistry`**, so users can inspect, customize, replace, disable, or extend how the agent reacts to Computer Vision events.

---

## 1. Built-In Micro Tools

| Tool Name | Purpose | Key Arguments |
|---|---|---|
| `send_alert` | Send an alert via callback, JSONL file, or webhook | `message`, `severity`, `camera_id`, `zone`, `channel`, `metadata` |
| `read_file` | Read a local file safely inside `workspace_root` | `path`, `max_bytes` |
| `write_file` | Create or overwrite a file inside `workspace_root` | `path`, `content`, `append=False` |
| `append_file` | Append a log entry to a file inside `workspace_root` | `path`, `content` |
| `list_files` | List workspace files (capped for low RAM) | `subdir=""`, `max_entries=50` |
| `db_connect` | Connect a local SQLite `.db` file under an alias | `db_path`, `alias="default"` |
| `db_list_tables` | List user tables in a connected SQLite database | `alias="default"` |
| `db_read_records` | Parameterized record lookup with row cap | `table`, `where={}`, `columns=None`, `limit=20`, `order_by=None`, `alias="default"` |
| `db_query` | Strictly read-only `SELECT` / `WITH` SQL query | `sql`, `params=[]`, `limit=20`, `alias="default"` |
| `db_insert_record` | Insert a parameterized dict row into a SQLite table | `table`, `record={}`, `alias="default"` |

---

## 2. Modifying & Customizing Tools for CV Events

Users have full runtime control over every tool:

```python
from phoenix.framework.cv_agent import CVAgent

agent = CVAgent(model="granite-1b-moe", workspace_root="./cv_edge")

# A) Add a new custom CV hardware tool (sync or async)
@agent.tool(name="trigger_siren", description="Sound local alarm siren for N seconds.")
def trigger_siren(zone: str, seconds: int = 5):
    print(f"SIREN ON in {zone} for {seconds}s")
    return {"siren": "triggered", "zone": zone, "seconds": seconds}

# B) Modify default arguments of an existing tool
agent.modify_tool("trigger_siren", default_args={"seconds": 12})

# C) Replace a built-in tool's implementation (e.g. route alerts to Telegram/MQTT)
async def custom_mqtt_alert(message: str, severity: str = "high", **kwargs):
    print(f"[MQTT:{severity}] {message} | extra={kwargs}")
    return {"sent": True, "transport": "mqtt"}

agent.modify_tool("send_alert", fn=custom_mqtt_alert)

# D) Temporarily disable a tool (e.g. prevent DB writes in audit-only mode)
agent.modify_tool("db_insert_record", enabled=False)

# E) Completely remove a tool
agent.remove_tool("append_file")

# F) Inspect current tools
for t in agent.list_tools():
    print(t["name"], "enabled=", t["enabled"], "defaults=", t["default_args"])
```

---

## 3. Sending Alerts with Anti-Spam Cooldown

CV detectors run at 10–60 FPS. Without deduplication, a single person standing in front of a camera would trigger hundreds of duplicate alerts.

`CVAgent` automatically tracks per-`(camera_id, zone, event_type)` cooldown windows via `CVAgentConfig(alert_cooldown_seconds=15.0)`:
- Duplicate `LOW` / `MEDIUM` / `HIGH` alerts within the cooldown window are suppressed (`context.alert_on_cooldown == True`).
- `CRITICAL` events (e.g. fire/weapon/fall emergencies) bypass cooldown so life-safety alerts are never dropped.

```python
agent = CVAgent(
    model="granite-1b-moe",
    config=CVAgentConfig(workspace_root="./cv_edge", alert_cooldown_seconds=30.0),
    on_alert=lambda payload: print("PUSH:", payload["severity"], payload["message"]),
    alert_log_path="./cv_edge/alerts.jsonl",      # Persistent local audit trail
    webhook_url=None,                             # Optional: "http://192.168.1.50:8080/hook"
)
```

---

## 4. Reading & Writing Local Files Safely

`LocalFileStore` enforces two edge-safety protections:
1. **Symlink-Safe Workspace Jail**: Paths like `../../etc/passwd` or symlinks pointing outside `workspace_root` raise `PermissionError`.
2. **RAM Byte Caps**: `max_read_bytes` (default 64 KB) and `max_write_bytes` (default 256 KB) guarantee a low-RAM phone can never OOM by accidentally reading a multi-gigabyte video file.

```python
# Direct Python calls (or invoked automatically by the <=1B MoE via tool_calls)
agent.write_file("reports/shift_01.txt", "Camera check OK\n")
agent.write_file("reports/shift_01.txt", "Gate north clear\n", append=True)

data = agent.read_file("reports/shift_01.txt")
print(data["content"])
```

---

## 5. Connecting to Local Databases & Reading Records

`LocalSQLiteStore` uses Python's built-in `sqlite3` (zero external packages, works out of the box on Android/Termux, iOS Python, Raspberry Pi, and Linux).

### Multiple Local Databases & Parameterized Record Reading
```python
# Connect primary and secondary local SQLite databases
agent.connect_database("./cv_edge/security.db", alias="default")
agent.connect_database("./cv_edge/inventory.db", alias="warehouse")

# Read records safely with parameterized WHERE filters & row cap
plates = agent.read_db_records(
    "authorized_plates",
    where={"status": "active", "zone": "north_gate"},
    columns=["plate", "owner", "shift"],
    limit=10,
)
print(plates["records"])

# Read-only SQL query (automatically blocks DROP/DELETE/UPDATE/ALTER/PRAGMA/multi-statement SQL)
rows = agent.db.query_readonly(
    "SELECT plate, owner FROM authorized_plates WHERE owner LIKE ?",
    params=["Ali%"],
    limit=10,
)
```

### Automatic Context Enrichment from Local DB (`set_db_enricher`)
Before the `<= 1B` MoE makes its decision, `set_db_enricher` injects matching SQLite records directly into the micro prompt (`DB_RECORDS:`) so even a 400M-active MoE can make accurate, grounded decisions:

```python
def lookup_vehicle(event, db):
    plate = event.metadata.get("plate")
    if not plate:
        return []
    return db.read_records("authorized_plates", where={"plate": plate}, limit=3)["records"]

agent.set_db_enricher(lookup_vehicle)
```
