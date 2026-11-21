# Phoenix AI (`phx-ashborn`) Agent Framework v1 — Complete Reference

Comprehensive developer documentation for building autonomous, tool-using, token-budgeted AI agents with `phoenix.framework.agent` and `phoenix.services.llm`.

---

## Documentation Map

| Guide | Covers | Primary Source Files |
| :--- | :--- | :--- |
| [01 — Agent Core, Profiles & Telemetry](01_agent_core_and_profiles.md) | `Agent`, `run()`, `run_stream()`, `resume()`, execution modes, `AgentProfile`, `RunMetrics`, `tracer` | [agent.py](../../phoenix/framework/agent/core/agent.py), [profile.py](../../phoenix/framework/agent/core/profile.py), [tracing.py](../../phoenix/services/observability/tracing.py) |
| [02 — LLM Services & Routing](02_llm_services_and_routing.md) | `BaseLLM`, `OpenAILLM`, `LocalLLM` (`transformers`, `llama_cpp`, `ollama`), `LLMRouter` (`MultiLLM` / `FallbackLLM`) | [base.py](../../phoenix/services/llm/base.py), [openai.py](../../phoenix/services/llm/openai.py), [local.py](../../phoenix/services/llm/local.py), [router.py](../../phoenix/services/llm/router.py) |
| [03 — Agent Loop & Cognition](03_agent_loop_and_cognition.md) | `AgentLoop`, `react` vs `plan` vs `fast_ans`, `Thinker`, `Planner`, `Actor`, `Reflector`, `Analyzer`, `CognitionPipeline`, schemas | [loop.py](../../phoenix/framework/agent/core/loop.py), [thinker.py](../../phoenix/framework/agent/cognition/thinker/thinker.py), [actor.py](../../phoenix/framework/agent/cognition/actor/actor.py), [schema.py](../../phoenix/framework/agent/cognition/schema.py) |
| [04 — Tools, Registry, Hooks & Runtime](04_tools_and_tool_registry.md) | `@tool`, `@agent.tool`, `BaseTool`, `FunctionTool`, schema inference, `ToolRegistry`, `ToolManager`, hooks, 12 default tools, 8 Tool Bank tools, `RestrictedPythonRuntime` | [base.py](../../phoenix/framework/agent/tools/base.py), [registry.py](../../phoenix/framework/agent/tools/registry.py), [tool_manager.py](../../phoenix/framework/agent/execution/tool_manager.py), [restricted_python.py](../../phoenix/framework/agent/runtime/restricted_python.py) |
| [05 — Memory & Context Management](05_memory_and_context_management.md) | 3-tier `MemoryManager` (`MemoryCell`, `ShortTermMemoryManager`, `LongTermMemoryManager`), `SessionMemory`, `ReflectionMemory`, token budgeting, checkpoints, `InteractiveMemoryAdapter` | [manager.py](../../phoenix/framework/agent/memory/manager.py), [cell.py](../../phoenix/framework/agent/memory/cell.py), [budget.py](../../phoenix/framework/agent/memory/budget.py), [adapter.py](../../phoenix/framework/agent/memory/adapter.py) |
| [06 — Scenarios & Recipes](06_scenarios_and_recipes.md) | Step-by-step patterns for ERP assistants, OpenRouter failover, offline GPU/CPU local LLMs, coding agents, HITL approvals, FastAPI SSE UIs, checkpoint recovery, multi-agent delegation | Cross-module end-to-end recipes |

---

## Architecture Overview

`phoenix.framework.agent` separates **LLM transport**, **cognition**, **memory**, **tool governance**, and **execution sandboxing** into swappable layers:

```
  User Prompt
       │
       ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │  Agent (core/agent.py)                                              │
  │   ├─ LLM Layer: OpenAILLM / LocalLLM / LLMRouter (services/llm)     │
  │   ├─ Profile:   AgentProfile (Identity, Role, Rules, tool_access)   │
  │   ├─ Memory:    MemoryManager (Working Cells + STM + LTM + Session) │
  │   ├─ Tools:     ToolRegistry + ToolManager (Hooks & Permissions)    │
  │   └─ Runtime:   RestrictedPythonRuntime (AST + Shell Sandbox)       │
  │                                                                     │
  │   ┌────────────────────────── AgentLoop ──────────────────────────┐ │
  │   │  mode="fast_ans"  -> Single LLM generation                    │ │
  │   │  mode="react"     -> Direct Think/Tool -> Observe -> Reply    │ │
  │   │  mode="plan"      -> Plan DAG -> Actor/Reflector -> Synthesis │ │
  │   └───────────────────────────────────────────────────────────────┘ │
  └─────────────────────────────────────────────────────────────────────┘
```

### Package Exports

Top-level lazy imports ([`phoenix/__init__.py`](../../phoenix/__init__.py)) keep cold import of `phoenix.framework.agent` fast without eagerly initializing vector DBs, audio, or RAG pipelines:

```python
from phoenix.framework.agent import (
    Agent,
    BaseTool,
    ToolResult,
    FunctionTool,
    CustomTool,      # alias of FunctionTool
    tool,
    create_tool,
    ensure_tool,
    ToolRegistry,
    LocalLLM,
    LLMRouter,
    MultiLLM,        # alias of LLMRouter
    FallbackLLM,     # alias of LLMRouter
)
```

---

## Installation & Requirements

### 1. System & Python Requirements

From [`pyproject.toml`](../../pyproject.toml) and [`setup.py`](../../setup.py):

- **Python**: `>= 3.9` (tested on Python 3.12).
- **Core dependencies** (always installed with `phx-ashborn`):
  - `python-dotenv>=1.0.0`
  - `requests>=2.28.0`
  - `openai>=1.0.0`
  - `pydantic`
  - `httpx>=0.24.0`
  - `numpy`

### 2. Optional Feature Extras

Install from the repository root (`-e .` for editable SDK development) or package name `phx-ashborn`:

| Extra | Command | Installed Libraries | Needed When |
| :--- | :--- | :--- | :--- |
| Core SDK only | `pip install -e .` | Core deps listed above | Using `Agent` + `OpenAILLM` / OpenRouter + `@tool` custom tools |
| `[agent]` | `pip install -e ".[agent]"` | `pydantic`, `redis>=4.5.0`, `celery>=5.3.0`, `duckduckgo-search`, `chromadb>=0.4.0`, `qdrant-client>=1.6.0` | Default `web_search` tool (`WebSearchTool`), Redis/Celery task stores |
| `[local]` | `pip install -e ".[local]"` | `torch`, `transformers`, `accelerate`, `bitsandbytes`, `sentence-transformers>=2.2.0` | Running HuggingFace models locally with `LocalLLM(..., backend="transformers")` |
| `[sensorium]` | `pip install -e ".[sensorium]"` | `psutil`, `pyserial>=3.5`, `paho-mqtt>=1.6.0` | Accurate RAM/CPU detection in `HardwareChecker` |
| `[full]` | `pip install -e ".[full]"` | Union of all extras | Full SDK stack |

**Additional Local Backends (not pinned in `setup.py`)**:
- **GGUF / `llama_cpp`**: Install `llama-cpp-python` (use a CUDA-compiled wheel if offloading layers to NVIDIA GPU).
- **Ollama**: Install and start the local Ollama server (`ollama serve`, default `http://localhost:11434`) and pull your model (`ollama pull qwen2.5-coder:7b`). Uses `httpx` already in core dependencies.
- **SQL Tool Bank (`SQLDatabaseTool`)**: Requires `sqlalchemy` (`pip install sqlalchemy`) plus your DB driver.
- **Web Scraper (`WebScraperTool`)**: Requires `beautifulsoup4` (`pip install beautifulsoup4`).

---

## Hardware Inspection (`HardwareChecker`)

Implemented in [`phoenix/core/hardware_check.py`](../../phoenix/core/hardware_check.py) and automatically consulted by `LocalLLM` when picking `device`, `n_threads`, and `n_gpu_layers`.

| Constant / Method | Signature / Value | Description |
| :--- | :--- | :--- |
| `MIN_RAM_GB` | `8` | Minimum total system RAM threshold (`ram["ok"] = total_ram >= 8`). |
| `RECOMMENDED_RAM_GB` | `16` | Recommended RAM threshold (`ram["recommended"]`). |
| `MIN_DISK_GB` | `10` | Minimum free disk space threshold before model downloads. |
| `HardwareChecker.check_all(silent=False)` | `-> Tuple[bool, Dict[str, Any]]` | Checks `ram`, `cpu`, `disk`, `os`, `gpu` and adds `recommendation`. Prints report unless `silent=True`. |
| `HardwareChecker.check_cpu()` | `-> Dict[str, Any]` | Returns `physical_cores`, `logical_cores`, `recommended_threads` (`min(physical, 16)`). |
| `HardwareChecker.check_ram()` | `-> Dict[str, Any]` | Returns `total_gb`, `available_gb`, `ok`, `recommended`. Falls back to `16.0 / 8.0` if `psutil` is not installed. |
| `HardwareChecker.check_disk()` | `-> Dict[str, Any]` | Returns `free_gb`, `ok` for current working directory. |
| `HardwareChecker.check_gpu(use_cache=True)` | `-> Dict[str, Any]` | Inspects CUDA/MPS if `torch` is already imported, else probes `nvidia-smi` (without forcing slow `import torch`), else Apple Silicon arm64 check. |
| `HardwareChecker.recommend_local_profile(results)` | `-> Dict[str, Any]` | Recommends `"4bit"` for CUDA `< 16 GB` VRAM (`"bf16"` if `>= 16 GB`), `"fp16"` for MPS, or `"8bit"` for CPU. |

---

## Global Configuration (`Config` & `.env`)

[`phoenix/core/config.py`](../../phoenix/core/config.py) runs `load_dotenv(override=True)` at import time and exposes `phoenix.core.config.config`. Key variables relevant to `Agent` & LLMs:

| Env Var / Attribute | Default | Used By |
| :--- | :--- | :--- |
| `OPENAI_API_KEY` | `""` | Default `api_key` in `OpenAILLM()` (falls back to mock mode if empty!) |
| `OPENAI_BASE_URL` | `"https://api.openai.com/v1"` | Default `base_url` in `OpenAILLM()` (override for OpenRouter, Groq, DeepSeek, etc.) |
| `OPENAI_LLM_MODEL` | `"gpt-4o"` | Fallback `model` if `OpenAILLM(model=None)` (note: if env is unset and `Config` has `"gpt-4o"`, `OpenAILLM` uses `Config.OPENAI_LLM_MODEL`) |
| `LOCAL_LLM_TEXT_MODEL` | `""` | Default model if `LocalLLM(model=None)` (else `"Qwen/Qwen2.5-3B-Instruct"`) |
| `LOAD_LOCAL_LLM` | `"false"` | Controls `LocalLLM.init()` when instantiated *without* explicit `model` or `gguf_filename` |
| `SECURITY_MAX_OUTPUT_LENGTH` | `2048` | Default `max_tokens` when callers do not pass an explicit `max_tokens` |
| `LOG_LEVEL` | `"INFO"` | Telemetry & SDK logger verbosity |

> [!CAUTION]
> **`load_dotenv(override=True)` Behavior**: Simply importing `phoenix.core.config` reads `.env` in your current working directory and **overwrites matching process environment variables**. Always set `api_key=...` and `base_url=...` explicitly in `OpenAILLM(...)` in multi-provider apps rather than relying only on global env vars.

---

## Minimal Quick Start

```python
import asyncio
from phoenix.framework.agent import Agent, tool
from phoenix.services.llm.openai import OpenAILLM

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b

async def main():
    llm = OpenAILLM(
        api_key="YOUR_API_KEY",
        base_url="https://openrouter.ai/api/v1",
        model="openai/gpt-4o-mini",
    )
    # Passing tools=[multiply] enables ONLY your custom tool (no default shell/file tools).
    agent = Agent(llm=llm, tools=[multiply])
    answer = await agent.run("What is 347 * 29?", mode="react")
    print(answer)
    print("Metrics:", agent.get_last_run_metrics())

asyncio.run(main())
```

---

## Global Requirements & Cautions Checklist

> [!IMPORTANT]
> 1. **Async Execution**: `agent.run()`, `agent.run_stream()`, `agent.resume()`, and all LLM methods (`generate`, `generate_structured`, `generate_with_tools`, `generate_stream`) are **coroutines / async generators** — always call them inside `asyncio.run(...)` or an `async def` context (such as FastAPI).
> 2. **Tool Loading Default Trap**: Passing `tools=None` (omitting `tools`) loads **all 12 default system tools** including `execute_command`, `python_repl`, and `file_delete`! Always pass `tools=[...]` (or `tools=[]`) unless you explicitly want filesystem/shell access, or set `include_default_tools=False` (default when `tools` is a list).
> 3. **Missing API Key Behavior**: If `OpenAILLM()` is instantiated with no `api_key`, `init()` logs a warning and enters **mock mode** (`generate()` returns `"[Mock OpenAI Response (No API Key) to: ...]"` rather than raising at init time), whereas `generate_structured()`/`generate_with_tools()` will fail when trying to call `self.client` if `self.client is None`.
> 4. **Fast Mode with Local Models**: When `llm` is a `LocalLLM` (or an `LLMRouter` whose providers are all `LocalLLM`), `Agent` automatically sets `fast_mode=True`, routing `mode="auto"` directly to `react` (with tool calls) or `fast_ans` (for casual greetings / no tools) and enabling `fast_reflect=True` (caps retries to `2` attempts and skips extra Reflector LLM calls on clean task successes).
> 5. **Isolated Sub-Projects**: Avoid calling `phoenix.main.init_phoenix()` when you only want `Agent` — `init_phoenix()` initializes caches, Chroma/Qdrant, embeddings, VLM, audio, and fine-tuning services globally. Construct `Agent(...)` directly with your chosen LLM, tools, and memory.
