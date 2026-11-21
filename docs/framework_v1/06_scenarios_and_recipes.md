# 06 — Scenarios & Production Recipes

Different real-world situations showing exact recommended configurations of `Agent`, `OpenAILLM`, `LocalLLM`, `LLMRouter`, tools, memory, hooks, and checkpoints.

---

## Situation 1: Domain / ERP Tool Assistant (Fast ReAct Backend Pattern)

Best for: ERP operations, CRM lookup/mutation, customer support APIs, dashboards (including backends like `Projects/erp_agent`).

**Why this configuration?**
- Passes `tools=[...]` explicitly (`include_default_tools=False`) so the agent **cannot** touch host files or execute shell commands.
- Locks permissions with `AgentProfile.tool_access`.
- Uses `mode="react"` + `max_iterations=6` so tool calls complete in **1–2 LLM round-trips** instead of full Plan/Reflect/Synthesize overhead.

```python
import asyncio
from typing import Dict
from phoenix.framework.agent import Agent, tool
from phoenix.framework.agent.core.profile import AgentProfile, Identity, Role, Personality
from phoenix.services.llm.openai import OpenAILLM

ERP_CUSTOMERS: Dict[str, dict] = {
    "C1": {"name": "Ahmed Mansour", "credit_limit": 4000.0, "balance_due": 1200.0},
    "C2": {"name": "Mona El-Sayed", "credit_limit": 10000.0, "balance_due": 1868.0},
}

@tool(read_only=True)
def check_customer_credit(customer_id: str) -> dict:
    """Return customer credit limit, balance due, and available credit."""
    cust = ERP_CUSTOMERS.get(customer_id)
    if not cust:
        return {"error": f"Customer '{customer_id}' not found"}
    return {
        "customer_id": customer_id,
        **cust,
        "available_credit": cust["credit_limit"] - cust["balance_due"],
    }

@tool
def post_payment(customer_id: str, amount: float) -> dict:
    """Record a positive customer payment and update balance due."""
    if amount <= 0 or customer_id not in ERP_CUSTOMERS:
        return {"error": "Invalid customer_id or non-positive payment"}
    ERP_CUSTOMERS[customer_id]["balance_due"] = max(
        0.0, ERP_CUSTOMERS[customer_id]["balance_due"] - amount
    )
    return {"customer_id": customer_id, "new_balance": ERP_CUSTOMERS[customer_id]["balance_due"]}

profile = AgentProfile(
    identity=Identity(name="NovaERP", id="erp-react-v1"),
    role=Role(title="ERP Operations Agent", mission="Answer ERP queries using verified tool data."),
    personality=Personality(communication_tone="Concise and factual", response_style="Bullet points"),
    rules=[
        "Always check credit or records via tools before stating numbers.",
        "Never retry failed write operations with the exact same arguments.",
    ],
    tool_access=["check_customer_credit", "post_payment"],
)

async def main():
    llm = OpenAILLM(
        api_key="YOUR_KEY",
        base_url="https://openrouter.ai/api/v1",
        model="openai/gpt-4o-mini",
    )
    agent = Agent(llm=llm, tools=[check_customer_credit, post_payment], profile=profile)
    reply = await agent.run(
        "Check available credit for C1 and explain if they can place a $3000 order.",
        session_id="erp_session_1",
        mode="react",
        max_iterations=6,
    )
    print(reply)
    print("Telemetry:", agent.get_last_run_metrics())

asyncio.run(main())
```

---

## Situation 2: High-Availability Multi-Provider / Multi-Key Routing

Best for: Free-tier rate limits (`429`), multi-region OpenAI keys, or cloud-first with local Ollama failover.

```python
import asyncio
from phoenix.framework.agent import Agent, LLMRouter, LocalLLM, tool
from phoenix.services.llm.openai import OpenAILLM

@tool(read_only=True)
def ping_service(name: str) -> str:
    """Check health of an internal microservice."""
    return f"{name}: OK (12ms)"

async def main():
    # Mix same-endpoint fallback_models + distinct keys/providers + Local fallback!
    router = LLMRouter(
        providers=[
            OpenAILLM(
                api_key="OPENROUTER_KEY_1",
                base_url="https://openrouter.ai/api/v1",
                model="qwen/qwen-2.5-coder-32b-instruct",
                fallback_models=["meta-llama/llama-3.3-70b-instruct:free"],
                cooldown_seconds=15.0,
            ),
            OpenAILLM(
                api_key="GROQ_KEY_2",
                base_url="https://api.groq.com/openai/v1",
                model="llama-3.3-70b-versatile",
            ),
            LocalLLM(model="qwen2.5-coder:7b", backend="ollama"),
        ],
        cooldown_seconds=25.0,
    )

    agent = Agent(llm=router, tools=[ping_service], fast_mode=True)
    res = await agent.run("Check billing-api health", mode="react")
    print(res)
    print("Failover history:", agent.get_last_run_metrics()["failover_events"])

asyncio.run(main())
```

> [!TIP]
> You can also skip instantiating `LLMRouter` directly by passing `llms=[...]` or `fallback_llm=...` into `Agent(...)` — `Agent._prepare_llm` builds `LLMRouter` automatically whenever `>= 2` LLM candidates are supplied.

---

## Situation 3: 100% Offline Local LLM (GPU 4-Bit vs CPU GGUF vs Ollama)

Check hardware first (`HardwareChecker.check_all()`), then match your hardware profile:

### Option A — Google Colab / NVIDIA GPU (`[local]` extras installed)
```python
from phoenix.framework.agent import Agent, LocalLLM

llm = LocalLLM(
    model="Qwen/Qwen2.5-3B-Instruct",
    backend="transformers",
    quantization="4bit",   # ~2.2 GB VRAM with bitsandbytes NF4 + SDPA
    device="cuda",
    n_ctx=4096,
    temperature=0.05,
)
# LocalLLM sets agent.fast_mode = True automatically!
agent = Agent(llm=llm, tools=[my_tool])
# answer = await agent.run("...", mode="auto")  # Auto-selects "react" when tools exist
```

### Option B — CPU Laptop / Low-RAM Server (GGUF via `llama-cpp-python` or `ollama`)
```python
# Recommended for CPU: Ollama or GGUF Q4_K_M (~2 GB RAM, fast C++ kernels)
llm_ollama = LocalLLM(model="qwen2.5:3b", backend="ollama", base_url="http://localhost:11434")

# OR direct GGUF from HuggingFace/disk:
llm_gguf = LocalLLM(
    model="Qwen/Qwen2.5-3B-Instruct-GGUF",
    gguf_filename="*q4_k_m.gguf",
    backend="llama_cpp",
    n_ctx=4096,
)
```

> [!CAUTION]
> Remember to clean up resources at shutdown: call `await llm.close()` for Ollama HTTP pools, and `LocalLLM.clear_cache()` if swapping multiple heavy local models in one process.

---

## Situation 4: Autonomous Coding & Repository Engineering Agent

Best for: Reading/searching/editing multi-file codebases safely inside a jailed workspace and running compilation/test checks.

```python
import asyncio
from phoenix.framework.agent import Agent
from phoenix.framework.agent.runtime.restricted_python import RestrictedPythonRuntime
from phoenix.framework.agent.core.loop import AgentLoop
from phoenix.services.llm.openai import OpenAILLM

async def main():
    workspace = "/abs/path/to/target_repo"
    runtime = RestrictedPythonRuntime(workspace_root=workspace, timeout=30.0)
    llm = OpenAILLM(api_key="YOUR_KEY", model="gpt-4o")

    agent = Agent(
        llm=llm,
        tools=None,                # Loads all 12 default file/code/search tools
        runtime=runtime,
        checkpoint_dir="./checkpoints",
        component_factories={
            "loop": lambda thinker, planner, actor, reflector, analyzer, memory: AgentLoop(
                thinker=thinker,
                planner=planner,
                actor=actor,
                reflector=reflector,
                analyzer=analyzer,
                memory_manager=memory,
                checkpoint_dir="./checkpoints",
                enable_workspace_analysis=True,   # Scans repo structure before planning
                parallel_tasks=True,
                max_concurrency=2,
            )
        },
    )

    result = await agent.run(
        f"Inspect {workspace}/calc.py, fix any edge-case bugs, and run compile_code on it.",
        session_id="repo_fix_01",
        mode="plan",
        max_iterations=12,
    )
    print(result)

asyncio.run(main())
```

---

## Situation 5: Human-in-the-Loop (HITL) Approval & Policy Hooks

Combines two independent safety layers:
1. **`ToolManager` before-hook** (`payload` inspection / blocking high-risk arguments silently).
2. **`AgentLoop.approval_callback`** (interactive user prompt returning `"approve"`, `"skip"`, or `"reject"`).

```python
from phoenix.framework.agent import Agent, ToolResult
from phoenix.framework.agent.core.loop import AgentLoop

def enforce_safe_sql(tool_name: str, payload: dict):
    if tool_name == "sql_database" and "drop " in str(payload.get("query", "")).lower():
        return ToolResult(success=False, output=None, error="Blocked DROP statement by policy hook.")
    return None  # Continue normally

async def human_gate(task, context: str) -> str:
    ans = input(f"\n[APPROVAL REQUIRED] Run '{task.description}'? [y/skip/no]: ").strip().lower()
    if ans in ("n", "no", "reject"):
        return "reject"
    if ans == "skip":
        return "skip"
    return "approve"

agent = Agent(
    llm=llm,
    tools=[sql_tool, refund_tool],
    component_factories={
        "loop": lambda thinker, planner, actor, reflector, memory: AgentLoop(
            thinker=thinker,
            planner=planner,
            actor=actor,
            reflector=reflector,
            memory_manager=memory,
            approval_callback=human_gate,
        )
    },
)
agent.tool_manager.add_before_hook(enforce_safe_sql)
```

---

## Situation 6: Streaming FastAPI Backend for a Web UI (e.g. SSE / WebSocket)

Ideal when wiring a custom isolated backend (like `Projects/erp_agent`) to a frontend UI using **`agent.run_stream()`** + **`on_event`**:

```python
import asyncio
import json
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from phoenix.framework.agent import Agent

app = FastAPI()

@app.post("/agent/stream")
async def stream_agent(body: dict):
    prompt = body["prompt"]
    session_id = body.get("session_id", "default")
    mode = body.get("mode", "react")
    trace_queue: asyncio.Queue = asyncio.Queue()

    async def on_event(ev: dict):
        await trace_queue.put({"type": "trace", **ev})

    agent = Agent(llm=my_llm, tools=my_erp_tools, on_event=on_event)

    async def event_stream():
        async for item in agent.run_stream(prompt, session_id=session_id, mode=mode, max_iterations=6):
            while not trace_queue.empty():
                yield f"data: {json.dumps(trace_queue.get_nowait())}\n\n"
            yield f"data: {json.dumps(item)}\n\n"
        while not trace_queue.empty():
            yield f"data: {json.dumps(trace_queue.get_nowait())}\n\n"
        yield f"data: {json.dumps({'type': 'metrics', 'data': agent.get_last_run_metrics()})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
```

---

## Situation 7: Long-Term Persistent Memory & Checkpoint Recovery

Persist cross-session user preferences/facts in JSONL and resume interrupted DAG runs:

```python
from phoenix.framework.agent import Agent
from phoenix.framework.agent.memory import MemoryManager

memory = MemoryManager(
    max_context_tokens=1800,
    stm_window=12,
    ltm_max_items=200,
    ltm_storage_path="./data/erp_ltm.jsonl",
)

agent = Agent(
    llm=llm,
    memory=memory,
    tools=[...],
    checkpoint_dir="./data/checkpoints",
)

# Seed persistent domain rule (pinned=True boosts retrieval across all sessions)
# await memory.add_context_item("org_1", "vat_rate", "14% Egypt standard VAT", long_term=True, pinned=True)

# On first run (e.g. interrupted after Task 1 of 3):
# await agent.run("Generate quarterly report...", session_id="q4_job", mode="plan")

# After process restart, reload memory & resume from checkpoint without redoing completed tasks:
# answer = await agent.resume(session_id="q4_job", max_iterations=10)
```

---

## Situation 8: Multi-Agent Coordinator (Sub-Agent Delegation via `@tool`)

Because every `Agent` is an ordinary async Python object, you can encapsulate specialized sub-agents (each with strict `AgentProfile` & minimal tools) as `@tool` functions on a coordinator `Agent`:

```python
from phoenix.framework.agent import Agent, tool

finance_agent = Agent(llm=llm, tools=[check_customer_credit, list_invoices])
warehouse_agent = Agent(llm=llm, tools=[check_inventory, reserve_stock])

@tool
async def ask_finance_specialist(question: str) -> str:
    """Delegate credit, billing, and invoice checks to the Finance Agent."""
    return await finance_agent.run(question, session_id="sub_finance", mode="react", max_iterations=4)

@tool
async def ask_warehouse_specialist(question: str) -> str:
    """Delegate SKU stock lookups and warehouse checks to the Warehouse Agent."""
    return await warehouse_agent.run(question, session_id="sub_warehouse", mode="react", max_iterations=4)

coordinator = Agent(llm=llm, tools=[ask_finance_specialist, ask_warehouse_specialist])
# coordinator decides whether to run sub-tasks sequentially or in parallel (`mode="plan"`),
# or quickly (`mode="react"`).
```
