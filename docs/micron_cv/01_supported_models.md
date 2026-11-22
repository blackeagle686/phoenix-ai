# Supported Models (`<= 1B` MoE Policy)

`CVAgent` enforces a **hard hardware safety policy** (`HARD_MAX_MOE_PARAMS_B = 1.0`) designed specifically for **mobile phone CPUs, ARM SBCs (Raspberry Pi / Orange Pi), and very low-RAM edge devices**.

Only **Mixture-of-Experts (MoE)** models at or below **1.0B parameters** are allowed. Any model exceeding `1.0B` or any non-MoE dense model is rejected immediately with `MicroMoEPolicyError` before allocating any memory.

---

## Supported Built-In Model IDs (Official Presets)

### 1. Downloadable Pre-Trained MoE Models (`<= 1B` Class)

| Model ID (Canonical) | Supported Aliases | Total / Active Params | Experts (Active) | Auto GGUF Repo (`llama_cpp`) | Est. Q4 RAM |
|---|---|---|---|---|---|
| `ibm-granite/granite-3.1-1b-a400m-instruct` | `granite-1b-moe`, `granite-moe-1b`, `granite-400m`, `granite-3.1-1b-a400m`, `micro-granite` | **1.0B** class / **0.4B** (400M) active | 32 (8 active/tok) | `bartowski/granite-3.1-1b-a400m-instruct-GGUF` (`*Q4_K_M.gguf`) | ~780 MB |
| `ibm-granite/granite-3.0-1b-a400m-instruct` | `granite-3.0-1b-moe`, `granite-3.0-400m` | **1.0B** class / **0.4B** (400M) active | 32 (8 active/tok) | `bartowski/granite-3.0-1b-a400m-instruct-GGUF` (`*Q4_K_M.gguf`) | ~780 MB |

### 2. Local / Custom Sub-1B MoE Preset Profiles (Bring Your Own GGUF or Callable)

Use these preset IDs when supplying a local quantized `.gguf` file via `gguf_path="..."` or a custom inference callable via `generate_fn=...`:

| Preset ID | Supported Aliases | Total / Active Params | Experts (Active) | Est. Q4 RAM |
|---|---|---|---|---|
| `micro-moe-500m-a150m` | `micro-moe-500m`, `tiny-moe-0.5b`, `edge-moe-500m` | **0.5B** / **0.15B** (150M) active | 8 (2 active/tok) | ~340 MB |
| `micro-moe-300m-a100m` | `micro-moe-300m`, `nano-moe-300m`, `phone-moe-300m` | **0.3B** / **0.10B** (100M) active | 8 (2 active/tok) | ~220 MB |

### 3. Custom Local GGUF or HuggingFace `<= 1B` MoE Models

You may load **any** custom local `.gguf` or HuggingFace MoE model as long as:
1. It is an MoE architecture (contains `moe` / `a400m` / `4x200m` in name, has `num_local_experts > 1` in local `config.json`, or passes `is_moe=True`), **AND**
2. Both total and active parameter counts are `<= 1.0B` (inferred from filename or passed explicitly via `params_b <= 1.0`).

```python
from phoenix.framework.cv_agent import CVAgent, MicroMoELLM

# Custom quantized local GGUF MoE (0.8B total = 4x200M)
llm = MicroMoELLM(
    model="./models/edge-4x200m-moe-q4_k_m.gguf",
    backend="llama_cpp",
    n_ctx=1024,
    n_threads=4,
)
agent = CVAgent(model=llm)
```

### 4. Zero-Model Ultra-Edge Mode (`model=None`)

For ultra-constrained devices (`<= 512 MB` RAM, battery-powered IoT cameras), pass `model=None`. `CVAgent` runs with **< 2 MB Python RAM** using deterministic `ContextDecisionRule`s, SQLite enrichment, alerts, and local file logging:

```python
agent = CVAgent(model=None, workspace_root="./cv_edge", db_path="./cv_edge/local.db")
```

---

## Explicitly Rejected Models (`MicroMoEPolicyError`)

`validate_micro_moe_model()` and `CVAgent(...)` **always reject** the following:

| Rejected Category | Examples | Reason |
|---|---|---|
| **MoE models `> 1.0B`** | `ibm-granite/granite-3.1-3b-a800m-instruct`, `allenai/OLMoE-1B-7B-0924-Instruct`, `Qwen/Qwen1.5-MoE-A2.7B-Chat`, `mistralai/Mixtral-8x7B-Instruct-v0.1`, `deepseek-ai/deepseek-moe-16b-chat`, `custom-moe-1.5b.gguf` | Exceeds `HARD_MAX_MOE_PARAMS_B = 1.0`. |
| **Non-MoE (Dense) models** | `Qwen/Qwen2.5-0.5B-Instruct`, `TinyLlama-1B`, `smollm-360m.gguf`, `Llama-3.2-1B` | Dense models are not MoE architectures. |
| **Unknown size without metadata** | `my-custom-moe.gguf` (without `params_b <= 1.0`) | Cannot verify compliance with the `<= 1.0B` RAM ceiling. |

```python
from phoenix.framework.cv_agent import CVAgent, MicroMoEPolicyError

try:
    CVAgent(model="Qwen/Qwen1.5-MoE-A2.7B-Chat")
except MicroMoEPolicyError as err:
    print("Blocked safely before RAM allocation:", err)
```

---

## Phone CPU & Low-RAM Runtime Tuning

`MicroMoELLM` configures conservative mobile-friendly defaults automatically:

| Parameter | Default | Hard Cap / Notes |
|---|---|---|
| `backend` | `"auto"` | Prefers `llama_cpp` (quantized GGUF Q4_K_M on CPU) when installed, falls back to `transformers` CPU `bfloat16` (`low_cpu_mem_usage=True`). |
| `n_ctx` | `1024` | Hard-capped at `2048` to prevent large KV-cache RAM spikes. |
| `max_tokens` | `160` | Hard-capped at `384` (micro-decisions only -> fast CPU latency). |
| `n_threads` | `min(4, cpu_count)` | Caps threads to `<= 4` to avoid phone thermal throttling. |
| `use_mmap` / `use_mlock` | `True` / `False` | Memory-maps GGUF weights without pinning them in scarce mobile RAM. |

```python
from phoenix.framework.cv_agent import CVAgent, MicroMoELLM

# Recommended phone/edge CPU setup with llama-cpp-python installed
llm = MicroMoELLM(
    model="ibm-granite/granite-3.1-1b-a400m-instruct",
    backend="llama_cpp",
    n_ctx=1024,
    max_tokens=140,
    n_threads=4,
    temperature=0.05,
)
agent = CVAgent(model=llm, workspace_root="./cv_workspace")
print(agent.get_stats()["model"])
```
