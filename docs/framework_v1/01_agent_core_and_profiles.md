# 01 — Agent Core, Profiles & Telemetry

Covers:
- [`phoenix/framework/agent/core/agent.py`](../../phoenix/framework/agent/core/agent.py) (`Agent`)
- [`phoenix/framework/agent/core/profile.py`](../../phoenix/framework/agent/core/profile.py) (`AgentProfile`, `Identity`, `Role`, `Personality`)
- [`phoenix/services/observability/tracing.py`](../../phoenix/services/observability/tracing.py) (`RunMetrics`, `Telemetry`, `tracer`)

---

## 1. `Agent` Class

`Agent` is the top-level orchestrator that binds together LLM provider(s)/routing, memory, safety runtime, tools, profile constraints, telemetry, cognition modules, and the execution loop.

```python
from phoenix.framework.agent import Agent
```

### 1.1 `Agent.__init__` Parameters

Every parameter is optional; sensible defaults are constructed automatically if omitted.

| Parameter | Type | Default | Detailed Behavior |
| :--- | :--- | :--- | :--- |
| `llm` | `Any` (e.g., `BaseLLM`, `OpenAILLM`, `LocalLLM`, `LLMRouter`, or `dict`) | `None` | Primary LLM provider. If `dict`, converted via `OpenAILLM(**llm)`. If all LLM params are `None`, defaults to `OpenAILLM()`. |
| `llms` | `Optional[Iterable[Any]]` | `None` | Additional ordered candidate LLMs/dicts. When more than 1 total LLM candidate exists across `llm`, `llms`, `fallback_llm`, `fallback_llms`, `Agent` wraps them automatically in `LLMRouter(candidates)`. |
| `fallback_llm` | `Optional[Any]` | `None` | Single backup LLM appended after `llm`/`llms` in auto-constructed `LLMRouter`. |
| `fallback_llms` | `Optional[Iterable[Any]]` | `None` | Multiple backup LLMs appended to `LLMRouter` candidates. |
| `memory` | `Optional[Any]` | `None` | Defaults to `MemoryManager()`. Custom memory objects missing `(add_interaction, get_full_context)` are automatically wrapped in `InteractiveMemoryAdapter(memory)`. |
| `tools` | `None \| list \| tuple \| set \| BaseTool \| Callable \| ToolRegistry` | `None` | **Careful:** If `None`, loads all 12 default tools (`ToolRegistry.load_default`). If a sequence (`list`/`tuple`/`set`) or single tool/callable, creates a clean registry with **only** those tools (unless `include_default_tools=True`). If a `ToolRegistry` instance, uses it directly and binds `self.runtime`. |
| `custom_tools` | `Optional[Iterable[Any]]` | `None` | Extra tools (`@tool`, `BaseTool`, callable, or tool dict) registered in addition to `tools`. Also auto-added to `profile.tool_access` if `profile.tool_access` is non-empty. |
| `include_default_tools` | `bool` | `False` | Only applies when `tools` is a collection (`list`/`tuple`/`set`) or a single tool/callable. When `True`, starts from `ToolRegistry.load_default()` before registering `tools`/`custom_tools`. Ignored when `tools=None` (defaults are always loaded) or when passing an existing `ToolRegistry`. |
| `thinker` | `Optional[Thinker]` | `None` | Custom `Thinker` override (or `component_factories["thinker"]`). |
| `planner` | `Optional[Planner]` | `None` | Custom `Planner` override. |
| `analyzer` | `Optional[Analyzer]` | `None` | Custom `Analyzer` override (used if `loop.enable_workspace_analysis=True` in `plan` mode). |
| `actor` | `Optional[Actor]` | `None` | Custom `Actor` override. |
| `reflector` | `Optional[Reflector]` | `None` | Custom `Reflector` override. |
| `tool_manager` | `Optional[ToolManager]` | `None` | Custom `ToolManager` override (holds before/after tool hooks & access filters). |
| `runtime` | `Optional[BaseRuntime]` | `None` | Defaults to `RestrictedPythonRuntime()`. Propagated to `ToolRegistry` and all registered tools missing a runtime. |
| `loop` | `Optional[AgentLoop]` | `None` | Explicit pre-built `AgentLoop` override. Bypasses `_construct_loop()`. |
| `profile` | `Optional[Union[AgentProfile, str, dict]]` | `None` | `AgentProfile` object, JSON file path `str` (`AgentProfile.from_json`), or `dict` (`AgentProfile.from_dict`). |
| `loop_cls` | `Optional[Type[AgentLoop]]` | `None` | Custom subclass of `AgentLoop` instantiated when building the default loop. |
| `component_factories` | `Optional[Dict[str, Callable[..., Any]]]` | `None` | Dependency-injection factory map for overriding `"tool_manager"`, `"thinker"`, `"planner"`, `"reflector"`, `"actor"`, `"analyzer"`, `"loop"`, **or registering custom named components** (e.g. `"validator"`, `"memory_index"`). Factories are called using signature inspection with available context keys. |
| `checkpoint_dir` | `Optional[str]` | `None` | Directory path where `AgentLoop` writes `<session_id>.json` checkpoints in `plan` mode (needed for `agent.resume(session_id)`). |
| `on_event` | `Optional[Callable[[Dict[str, Any]], Any]]` | `None` | Sync or async callback invoked for real-time telemetry events (`react_start`, `plan`, `task_start`, `task_done`, `tool_call`, `replan`, `complete`). |
| `fast_mode` | `Optional[bool]` | `None` | When `None` (auto), sets `True` iff all configured LLMs are `LocalLLM`. Controls `auto` mode routing (`react`/`fast_ans` instead of LLM classifier) and `fast_reflect` in `AgentLoop`. |

> [!CAUTION]
> **Precedence in Component Construction**: Explicit constructor argument (`thinker=...`) **>** `component_factories["thinker"]` **>** default `Thinker(...)`.
> Also note: custom names in `component_factories` (anything other than `tool_manager`, `thinker`, `planner`, `analyzer`, `actor`, `reflector`, `loop`) are instantiated and attached to `self.<name>` AND passed as kwargs into `loop_cls` if accepted by its signature!

---

### 1.2 Execution Methods

#### `await agent.run(prompt, session_id=None, max_iterations=15, mode="auto") -> str`

Runs one turn/goal to completion and stores run telemetry in `self.last_run_metrics`.

- **Parameters**:
  - `prompt: str`: User request / task prompt.
  - `session_id: Optional[str] = None`: Conversation/session key. Generates a random `uuid4()` string if `None`. Re-using the same `session_id` preserves Short-Term Memory across turns!
  - `max_iterations: int = 15`: Iteration cap passed to `run_react` or `loop.run`.
  - `mode: str = "auto"`: Normalized by `_normalize_mode(mode)`:
    - `"fast_ans"` (aliases: `"fast_mode"`, `"fast"`, `"direct"`, `"chat"`): Single direct `llm.generate()` call using memory context. No tool calls.
    - `"react"`: Direct `AgentLoop.run_react()` (Think/Tool -> Observe -> Reply, typically 1-2 LLM calls).
    - `"plan"`: Full `AgentLoop.run()` (`PlanSchema` DAG -> `Actor` -> `Reflector` -> Synthesis, or legacy iterative action loop).
    - `"auto"`:
      - If `self.fast_mode` is `True`: Uses `"fast_ans"` if `_looks_like_casual_greeting(prompt)` OR `self.get_tools_info()` is empty; otherwise uses `"react"`.
      - If `self.fast_mode` is `False`: Makes an extra 10-token LLM classification call returning `"PLAN"` (`mode="plan"`) or `"FAST"` (`mode="fast_ans"`).

> [!WARNING]
> **Why `mode="auto"` with cloud LLMs does NOT trigger `react`**: When `fast_mode=False` (default for `OpenAILLM`), `mode="auto"` asks the LLM to classify `'PLAN'` vs `'FAST'`, so it **never selects `'react'`** automatically — it routes to either full `plan` DAG or `fast_ans`. If you want fast single-loop tool calling with a cloud LLM (e.g., an ERP assistant or chatbot with tools), pass **`mode="react"` explicitly** or set **`fast_mode=True`**!

#### `async for event in agent.run_stream(prompt, session_id=None, max_iterations=15, mode="auto")`

Async generator yielding progress & text chunk dicts:
- `{"type": "status", "content": "..."}`
- `{"type": "chunk", "content": "...\n"}`

Executes the same mode resolution as `run()`, routing to:
- Direct `fast_ans` chunked lines, OR
- `loop.run_react_stream(...)` when `actual_mode == "react"`, OR
- `loop.run_stream(...)` when present, else falling back to `loop.run(...)` chunked line-by-line.
Always populates `self.last_run_metrics` in `finally`.

#### `await agent.resume(session_id, max_iterations=15) -> str`

Resumes an interrupted **`plan`-mode** run saved under `checkpoint_dir/<safe_session_id>.json`.
- **Requirements**:
  1. `checkpoint_dir` must have been provided on `Agent(...)` or `AgentLoop(...)`.
  2. At least one `plan` checkpoint must exist on disk for `session_id` (otherwise raises `FileNotFoundError`).
- **Behavior**: Calls `loop.resume(self.memory, session_id, max_iterations=max_iterations)`, skipping all tasks already marked `done` in `active_memory.completed_tasks`.

#### `agent.get_last_run_metrics() -> Optional[Dict[str, Any]]`

Returns a copy of `self.last_run_metrics` from the most recent `run()`, `run_stream()`, or `resume()` call (see §3 below for exact keys).

---

### 1.3 Runtime Tool & Component Mutation Methods

| Method | Signature | Description |
| :--- | :--- | :--- |
| `register_tool` | `(tool_or_func, name=None, description=None, allow_in_profile=True, **kwargs) -> BaseTool` | Wraps and registers a `BaseTool`, callable, or dict into `self.tools`, binds `self.runtime`, and appends `name` to `profile.tool_access` when `allow_in_profile=True` and `profile.tool_access` is non-empty. |
| `register_tools` | `(tools: Iterable[Any], allow_in_profile=True) -> List[BaseTool]` | Batch version of `register_tool`. |
| `unregister_tool` | `(name: str) -> bool` | Removes a tool by name from `self.tools`. |
| `tool` (`@agent.tool`) | Decorator (`@agent.tool` or `@agent.tool("name", "desc", ...)`) | Defines and immediately registers a custom function on this `Agent` instance. |
| `get_tools_info` | `() -> List[Dict[str, Any]]` | Returns permitted tool schemas (`ToolManager.get_available_tools_info()`). |
| `set_component` | `(name: str, component: Any, rebuild_loop: bool = True)` | Swaps a component at runtime (core or custom). Calls `rebuild_loop()` unless `rebuild_loop=False` or `name == "loop"`. |
| `rebuild_loop` | `(loop_cls: Optional[Type[AgentLoop]] = None)` | Re-instantiates `self.loop` from current components (`loop_cls` optionally replaces `self.loop_cls`). |
| `set_cognition_pipeline` | `(bootstrap_pipeline_path=None, loop_pipeline_path=None)` | Configures a JSON `CognitionPipeline` spec on `self.loop`. |

---

## 2. `AgentProfile` System

Defined in [`phoenix/framework/agent/core/profile.py`](../../phoenix/framework/agent/core/profile.py). Injects structured identity, rules, capabilities, and enforces **tool access allowlisting**.

### 2.1 Pydantic Schema Structure

```python
from phoenix.framework.agent.core.profile import (
    AgentProfile, Identity, Role, Personality
)

profile = AgentProfile(
    identity=Identity(name="NovaERP", id="erp-agent-01"),
    role=Role(
        title="Autonomous ERP Operations Specialist",
        mission="Verify inventory, customer credit, and create compliant invoices."
    ),
    personality=Personality(
        communication_tone="Concise and audit-ready",
        response_style="Structured bullet points with exact IDs and totals"
    ),
    rules=[
        "Never fabricate invoice IDs or stock counts without calling tools.",
        "Stop immediately if credit_limit is exceeded and explain the shortfall."
    ],
    capabilities=["Inventory inspection", "Credit validation", "Invoice posting"],
    tool_access=["check_inventory", "check_customer_credit", "create_order_invoice"],
)
```

Also supports:
- `AgentProfile.from_json("path/to/profile.json")`
- `AgentProfile.from_dict(config_dict)`
- `profile.to_prompt_string()` (automatically appended to `Thinker` and `Analyzer` system prompts).

### 2.2 Tool Access Filtering Rules & Cautions

Look carefully at how `ToolManager.is_tool_allowed(tool_name)` (`phoenix/framework/agent/execution/tool_manager.py`) works:
1. If `profile is None` OR `profile.tool_access` is empty (`[]`): **ALL tools in `ToolRegistry` are allowed**.
2. If `"*"` is in `profile.tool_access`: **ALL tools are allowed**.
3. If `profile.tool_access` is a non-empty list without `"*"`: **ONLY listed tool names are permitted** (`get_available_tools_info()`, `get_openai_tools()`, `execute_tool()`, and `execute_tool_result()` all enforce this!).
4. **Interaction with `Agent(..., custom_tools=...)` / `register_tool()`**:
   - Tools passed to `Agent(tools=[...], custom_tools=[...])` or `agent.register_tool(..., allow_in_profile=True)` automatically append their `name` to `profile.tool_access` *only if `profile.tool_access` was already non-empty*!
   - **Caution**: If you create `Agent(..., tools=existing_registry, profile=strict_profile)` where `strict_profile.tool_access = ["only_tool_a"]`, tools already inside `existing_registry` other than `"only_tool_a"` will be blocked unless listed (or `allow_in_profile=True` via `register_tool`) — which is great for locking down an existing `ToolRegistry.load_default()`!

---

## 3. Telemetry, Streaming Events & `RunMetrics`

Implemented in [`phoenix/services/observability/tracing.py`](../../phoenix/services/observability/tracing.py) via `contextvars` (async concurrency safe per coroutine task!).

### 3.1 `RunMetrics` Schema (Returned by `agent.get_last_run_metrics()`)

```python
{
    "run_id": "a45aabee72f1",          # 12-char hex UUID
    "session_id": "...",
    "status": "success",               # "running" | "success" | "error"
    "duration_ms": 1842.35,
    "llm_calls": 3,                    # Incremented by OpenAILLM / LocalLLM / BaseLLM spans
    "prompt_tokens": 1420,
    "completion_tokens": 215,
    "total_tokens": 1635,
    "models_used": ["gpt-4o-mini"],    # Deduplicated ordered list
    "retries": 0,                      # Retry attempts in OpenAILLM._call_with_resilience
    "failovers": 0,                    # Provider/model switches logged via tracer.record_failover
    "failover_events": [               # [{"from": ..., "to": ..., "reason": ...}]
        ...
    ],
    "tool_calls": 2,                   # Recorded in ToolManager.execute_tool_result
    "tool_failures": 0,
    "tool_durations_ms": {             # Cumulative ms per tool name in this run
        "check_inventory": 4.21
    },
}
```

### 3.2 Real-Time `on_event` Callback

Pass `on_event=fn` (sync or `async def`) to `Agent(...)` to receive live lifecycle events via `await tracer.emit_event(...)`. Every payload contains `{"type": <str>, "timestamp": <float>, ...}`:

| `event["type"]` | Emitted By | Extra Fields |
| :--- | :--- | :--- |
| `"react_start"` | `AgentLoop.run_react` | `session_id`, `prompt` (first 120 chars) |
| `"tool_call"` | `AgentLoop.run_react` | `step`, `tool` |
| `"plan"` | `AgentLoop._run_plan_dag_execution` | `session_id`, `objective`, `tasks` (count) |
| `"task_start"` | `AgentLoop._run_plan_dag_execution` | `task_id`, `description` |
| `"task_done"` | `AgentLoop._run_plan_dag_execution` | `task_id`, `completed` (`bool`) |
| `"replan"` | `AgentLoop._run_plan_dag_execution` | `added_tasks` (count) |
| `"complete"` | `AgentLoop` (`react` & `plan`) | `session_id`, optionally `mode="react"` |

> [!CAUTION]
> **Custom `BaseLLM` Spans & Legacy `execute_tool`**:
> - `RunMetrics.add_llm_usage` only increments `llm_calls` and tokens when spans start with `"OpenAILLM."`, `"LocalLLM."`, `"BaseLLM."`, or `"LLMRouter."` and finish with `status="success"`. If you write a custom `BaseLLM` subclass, emit spans with `"BaseLLM.generate"` via `tracer.start_span`/`end_span`.
> - `tracer.record_tool(...)` is called inside `ToolManager.execute_tool_result(...)` (used by `Actor`), **not** inside legacy `ToolManager.execute_tool(...)`.
