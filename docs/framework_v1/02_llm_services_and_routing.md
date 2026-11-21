# 02 — LLM Services (`OpenAILLM`, `LocalLLM`) & Multi-Provider Routing (`LLMRouter`)

Covers:
- [`phoenix/services/llm/base.py`](../../phoenix/services/llm/base.py) (`BaseLLM`, `_extract_json_dict`)
- [`phoenix/services/llm/openai.py`](../../phoenix/services/llm/openai.py) (`OpenAILLM`)
- [`phoenix/services/llm/local.py`](../../phoenix/services/llm/local.py) (`LocalLLM`)
- [`phoenix/services/llm/router.py`](../../phoenix/services/llm/router.py) (`LLMRouter`, `MultiLLM`, `FallbackLLM`)

---

## 1. `BaseLLM` Interface Contract

Every LLM backend in Phoenix inherits from [`BaseLLM`](../../phoenix/services/llm/base.py):

| Method | Signature | Description |
| :--- | :--- | :--- |
| `is_available()` | `-> bool` | Readiness check before execution. |
| `generate(prompt, session_id=None, max_tokens=None)` | `async -> str` | Standard text completion. |
| `generate_stream(prompt, session_id=None, max_tokens=None)` | `async -> AsyncGenerator[str, None]` | Token/chunk streaming completion. |
| `generate_structured(prompt, schema, session_id=None, max_tokens=None)` | `async -> BaseModel \| Any` | Prompts for strict JSON matching Pydantic `schema`, extracts JSON via `_extract_json_dict()`, and validates with `schema.model_validate(...)` (or `schema(**parsed)`). |
| `generate_with_tools(prompt, tools, session_id=None, max_tokens=None)` | `async -> TaskExecutionSchema` | Default falls back to `generate_structured(..., TaskExecutionSchema)`. Subclasses (`OpenAILLM`, `LocalLLM` with Ollama) override with native tool-calling. |

**Important when writing a mock or custom `BaseLLM`**:
`Thinker._supports_native_tool_calling()` checks whether `type(self.llm).generate_with_tools` is overridden beyond `BaseLLM.generate_with_tools`. Thus, simple custom/mock LLMs that only implement `generate()` and/or `generate_structured()` automatically work cleanly with `Agent`!

---

## 2. `OpenAILLM` (Cloud & OpenAI-Compatible Endpoints)

Works with **OpenAI**, **OpenRouter**, **Groq**, **DeepSeek**, **Together**, **Fireworks**, **vLLM**, and **LM Studio** via `AsyncOpenAI` + `httpx.Timeout(120.0, connect=60.0)`.

```python
from phoenix.services.llm.openai import OpenAILLM
```

### 2.1 `OpenAILLM.__init__` Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `api_key` | `Optional[str]` | `config.OPENAI_API_KEY` | API key for target provider. If empty/missing, `init()` logs a warning and sets `client=None` (mock mode for `generate()`). |
| `base_url` | `Optional[str]` | `config.OPENAI_BASE_URL` | Base API URL (`"https://api.openai.com/v1"` by default; trailing `/` is stripped in `init()`). Set `"https://openrouter.ai/api/v1"` for OpenRouter, `"https://api.groq.com/openai/v1"` for Groq, etc. |
| `model` | `Optional[str]` | `config.OPENAI_LLM_MODEL` or `"LongCat-Flash-Chat"` | Primary model identifier. |
| `fallback_models` | `Optional[List[str]]` | `[]` | Same-endpoint fallback models rotated automatically on rate limits (`429`) or provider failures (ideal for OpenRouter free models!). |
| `max_retries` | `int` | `2` | Retries per model using exponential backoff (`min(8.0, retry_base_delay * (2 ** attempt) + jitter)`). Clamped `>= 0`. When `fallback_models` has another candidate left, attempts on the current model cap at `max(1, min(2, max_retries + 1))` and fail over immediately on `429`. |
| `retry_base_delay` | `float` | `0.5` | Base delay in seconds (clamped `>= 0.05`). |
| `cooldown_seconds` | `float` | `15.0` | Cooldown window in seconds after an HTTP `429` before putting a cooled model back ahead of fallback models (clamped `>= 1.0`). |

### 2.2 Methods & Special Features

- **`await llm.init()`**: Creates `AsyncOpenAI(api_key=..., base_url=..., timeout=httpx.Timeout(120.0, connect=60.0))`. Returns cleanly (`self.client = None`) when `api_key` is missing.
- **`llm.is_available() -> bool`**: True iff both `self.api_key` and `self.model` are truthy.
- **`await llm.generate(prompt, session_id=None, max_tokens=None, mode=None) -> str`**:
  - `mode` (or `config.OPENAI_GENERATION_MODE`, default `"standard"`):
    - `"smart"`, `"fast_smart"`, `"thinking"`, `"planning"` activate a reasoning system prompt, slice legacy history to last 8 items, and scan for inline **`calc: <expr>`** lines evaluated safely via AST arithmetic (`_safe_eval_math_expression`, supporting `+ - * / % **`).
  - If `session_id` is set and container has a `"memory"` service registered, reads/writes legacy `memory.history` / `memory.search_memory`. (Note: inside `Agent`, `Agent.run` handles memory separately and passes `session_id=None` for `fast_ans`.)
- **`await llm.generate_structured(prompt, schema, session_id=None, max_tokens=None)`**:
  - Sends `response_format={"type": "json_object"}` first; if the endpoint returns HTTP `400` or `422` (provider doesn't support `response_format`), retries once without `response_format` and extracts JSON via `_extract_json_dict`. **Does not** strip `response_format` on `429`/timeouts so rate limits cleanly trigger retries/failover.
- **`await llm.generate_with_tools(prompt, tools, session_id=None, max_tokens=None) -> TaskExecutionSchema`**:
  - Uses native `tools=tools, tool_choice="auto"`.
  - If the provider returns `400/404/422/501` complaining about `"tool"`, `"function"`, `"tool_choice"`, `"unsupported"`, or `"not support"`, caches `self.model` in `self._tools_unsupported` and transparently falls back to `generate_structured(..., TaskExecutionSchema)`!

### 2.3 Resilience Error Handling Rules

Implemented in `_is_retryable_exception` & `_call_with_resilience`:

| HTTP Error / Condition | Retry Same Model? | Cooldown Applied? | Fails Over to Next Model? |
| :--- | :--- | :--- | :--- |
| `429` / `"rate limit"` / `"too many requests"` | Yes (or skips fast if fallback exists) | Yes (`cooldown_seconds`) | Yes immediately |
| `500`, `502`, `503`, `504`, `timeout`, `connection` | Yes (up to attempts) | No (only `429` cools down) | Yes after attempts |
| `401`, `403`, `404`, `422` | No (breaks immediately) | Internal flag only | Yes immediately |
| `400` (bad request) | No | No | Yes if fallback exists |

> [!CAUTION]
> **Mock Mode Trap in `OpenAILLM`**:
> Calling `generate()` with no `api_key` returns a mock string `"[Mock OpenAI Response (No API Key) to: ...]"`. However, calling `generate_structured()` or `generate_with_tools()` without an `api_key` leaves `self.client = None`, causing an `AttributeError` wrapped in `RuntimeError("Structured generation failed: ...")`. Always check `llm.is_available()` or provide a real API key before running `Agent` with `OpenAILLM`.

---

## 3. `LocalLLM` (Ollama, `llama_cpp` / GGUF, HuggingFace `transformers`)

Supports offline inference across CPU, NVIDIA CUDA, and Apple Silicon MPS with process-wide model caching (`LocalLLM._model_cache`), thread-safe locking (`_inference_lock`), and native structured JSON / streaming.

```python
from phoenix.framework.agent import LocalLLM
```

### 3.1 `LocalLLM.__init__` Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `model` | `Optional[str]` | `config.LOCAL_LLM_TEXT_MODEL or "Qwen/Qwen2.5-3B-Instruct"` | HuggingFace repo ID (`"Qwen/Qwen2.5-3B-Instruct"`), local `.gguf` file path, GGUF repo ID (with `gguf_filename`), or Ollama tag (`"qwen2.5-coder:7b"`). |
| `backend` | `str` | `"auto"` | `"auto"`, `"llama_cpp"` (`llamacpp`, `llama-cpp`, `gguf`), `"transformers"` (`transformer`, `hf`, `huggingface`), or `"ollama"`. |
| `quantization` | `str` | `"auto"` | For `transformers`: `"auto"`, `"4bit"` (CUDA NF4 double-quant via `bitsandbytes`), `"8bit"` (CUDA 8-bit or CPU dynamic INT8 if RAM >= 10 GB), `"fp16"`, `"bf16"`, `"none"`. |
| `device` | `str` | `"auto"` | `"auto"`, `"cuda"`, `"mps"`, `"cpu"`. |
| `gguf_filename` | `Optional[str]` | `None` | Filename/glob when loading GGUF from a HuggingFace repo (e.g., `"*Q4_K_M.gguf"`). |
| `n_ctx` | `int` | `4096` | Context window size (clamped `>= 512`). |
| `n_gpu_layers` | `Optional[int]` | `None` | For `llama_cpp`: defaults to `-1` (all layers on GPU) when CUDA/MPS available & supported, else `0`. |
| `n_threads` | `Optional[int]` | `None` | CPU threads; defaults to `HardwareChecker.check_cpu()["recommended_threads"]`. |
| `temperature` | `float` | `0.1` | Sampling temperature (clamped `>= 0.0`). Values `<= 0.01` disable sampling in `transformers` (`do_sample=False`). |
| `top_p` | `float` | `0.9` | Nucleus sampling parameter. |
| `base_url` | `str` | `"http://localhost:11434"` | Ollama server URL (trailing `/` stripped). |
| `keep_alive` | `str` | `"30m"` | Ollama `keep_alive` retention duration. |
| `trust_remote_code` | `bool` | `False` | Passed to `AutoTokenizer` / `AutoModelForCausalLM`. |
| `allow_auto_load` | `Optional[bool]` | `None` | Defaults to `True` if `model` or `gguf_filename` was explicitly passed, else relies on `config.LOAD_LOCAL_LLM`. |

### 3.2 Backend Auto-Resolution & Quantization Matrix

When `backend="auto"`, `_resolve_backend()` chooses:
1. **`llama_cpp`**: if `gguf_filename` is set OR `model.lower().endswith(".gguf")`.
2. **`ollama`**: if `":" in model` and `"/" not in model` and `not os.path.exists(model)` (e.g., `"qwen2.5:7b"`).
3. **`transformers`**: otherwise.

> [!NOTE]
> **Fallback Behavior Difference**:
> - When `backend="auto"` and `llama_cpp` or `transformers` fails to load, `LocalLLM.init()` **logs a warning and falls back to `ollama`**.
> - When you specify an explicit `backend="llama_cpp"` or `backend="transformers"`, initialization failures **raise `RuntimeError` loudly** so misconfigurations are never silently hidden!

#### Memory & Hardware Matrix (`transformers`)

| Device | Quantization Requested | Actual Behavior in `_load_transformers_sync` |
| :--- | :--- | :--- |
| **`cuda`** | `"auto"` / `"4bit"` | Attempts `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)` + `device_map="auto"` + `attn_implementation="sdpa"`. If `bitsandbytes` missing, falls back to `bfloat16`/`float16`. |
| **`cuda`** | `"8bit"` | `BitsAndBytesConfig(load_in_8bit=True)`. |
| **`cuda`** | `"bf16"` / `"fp16"` | Loads in requested `torch_dtype` (`sdpa` attention). |
| **`mps`** | any | `torch.float16` (`device_map="mps"` if `accelerate` installed). |
| **`cpu`** | `"auto"` / `"4bit"` / `"bf16"` | Loads in **`torch.bfloat16`** (50% less RAM than fp32 — prevents OOM!). |
| **`cpu`** | `"8bit"` | Loads fp32 then applies `torch.ao.quantization.quantize_dynamic` **ONLY IF `available_gb >= 10.0`** (prevents OOM spike during conversion!). |

### 3.3 `LocalLLM` Methods

- **`generate(prompt, session_id=None, max_tokens=None, temperature=None, system_prompt=None) -> str`**
- **`generate_structured(prompt, schema, session_id=None, max_tokens=None)`**:
  - Forces `temperature=0.0`.
  - Uses `response_format={"type": "json_object"}` in `llama_cpp` (avoids slow Pydantic `$defs` GBNF compilation).
  - Passes JSON schema `format=schema_dict` in Ollama.
  - Performs **one deterministic repair attempt** (capped at `min(out_tokens, 1024)`) if `_extract_json_dict(raw)` returns empty!
- **`generate_with_tools(prompt, tools, session_id=None, max_tokens=None) -> TaskExecutionSchema`**:
  - On **Ollama**: sends native `/api/chat` `tools` first; if rejected, caches model in `_ollama_tools_unsupported` and switches to compact JSON prompt injection + `generate_structured`.
  - On **`transformers` / `llama_cpp`**: injects compact tool JSON via `_compact_tools_for_prompt(tools)` and calls `generate_structured(..., TaskExecutionSchema)`.
- **`generate_stream(...)`**: Async token streaming across Ollama (`/api/chat` `stream: true`), `llama_cpp` (`create_chat_completion(..., stream=True)` over `asyncio.Queue`), and `transformers` (`TextIteratorStreamer`).
- **`await llm.close()`**: Closes pooled `httpx.AsyncClient` (Ollama).
- **`LocalLLM.clear_cache()`**: Clears `LocalLLM._model_cache`, runs `gc.collect()`, and calls `torch.cuda.empty_cache()` if CUDA is available.

### 3.4 `LocalLLM` Requirements & Cautions

> [!CAUTION]
> 1. **Colab / Cloud Notebook CPU Wheels**: If `llama-cpp-python` was compiled for CPU-only without CUDA flags, `LocalLLM` emits a warning that prompt evaluation will run on CPU. On CUDA GPUs (e.g., T4/L4), prefer HuggingFace `backend="transformers"` + `quantization="4bit"` (`pip install bitsandbytes accelerate`) OR install a CUDA-enabled `llama-cpp-python` wheel.
> 2. **Incompatible Old CUDA GPUs**: Before loading onto CUDA, `_load_transformers_sync` runs `torch.zeros(1, device="cuda") + 1`. Older GPUs (e.g., `sm_61` MX330 with modern PyTorch requiring `>= sm_75`) gracefully fall back to CPU with a warning rather than crashing mid-load.
> 3. **Process-Wide Cache & `n_ctx` Limits**: Models are keyed by `f"{backend}:{model}:{gguf_filename}:{quantization}:{device}:{n_ctx}"`. Changing `n_ctx` loads a second copy into RAM/VRAM unless you call `LocalLLM.clear_cache()` first.
> 4. **CPU 4-bit Reality**: PyTorch `transformers` + `bitsandbytes` NF4 4-bit only works on CUDA; on CPU, `quantization="4bit"` safely loads `bfloat16`. For true 4-bit ~2 GB RAM on CPU, use a **GGUF** model (`backend="llama_cpp"`) or **Ollama**.

---

## 4. `LLMRouter` (`MultiLLM` / `FallbackLLM`)

Resilient multi-provider, multi-key, and hybrid cloud+local failover router ([`phoenix/services/llm/router.py`](../../phoenix/services/llm/router.py)). `MultiLLM` and `FallbackLLM` are aliases for `LLMRouter`.

### 4.1 Three Ways to Configure Routing

#### Option A — Distinct Providers / API Keys / Hybrid Cloud + Local
```python
from phoenix.framework.agent import Agent, LLMRouter, LocalLLM
from phoenix.services.llm.openai import OpenAILLM

router = LLMRouter(
    providers=[
        OpenAILLM(api_key="KEY_1", base_url="https://api.groq.com/openai/v1", model="llama-3.3-70b-versatile"),
        OpenAILLM(api_key="KEY_2", base_url="https://openrouter.ai/api/v1", model="qwen/qwen-2.5-coder-32b-instruct"),
        LocalLLM(model="qwen2.5:7b", backend="ollama"),
    ],
    cooldown_seconds=20.0,
)
agent = Agent(llm=router, tools=[...])
```
You can also pass `dict` configs inside `providers`:
```python
router = LLMRouter([
    {"api_key": "k1", "base_url": "https://openrouter.ai/api/v1", "model": "model-a:free"},
    {"api_key": "k2", "base_url": "https://api.openai.com/v1", "model": "gpt-4o-mini"},
])
```

#### Option B — `LLMRouter.from_models(...)` (One Key, Multiple Models)
```python
router = LLMRouter.from_models(
    api_key="OPENROUTER_KEY",
    base_url="https://openrouter.ai/api/v1",
    models=[
        "meta-llama/llama-3.3-70b-instruct:free",
        "qwen/qwen-2.5-72b-instruct:free",
        "google/gemma-2-9b-it:free",
    ],
    cooldown_seconds=30.0,
    max_retries=1,
)
```

#### Option C — Directly via `Agent(...)` Shorthand
```python
agent = Agent(
    llm=OpenAILLM(api_key="k1", model="primary", fallback_models=["backup-1"]),
    fallback_llm=LocalLLM(model="qwen2.5:7b", backend="ollama"),
)
```

### 4.2 Routing & Cooldown Mechanics

- **`_ordered_indices()`**: Evaluates all healthy providers first (in list order), then cooled providers at the back as a last resort so all options are still exhausted before raising.
- **Cooldown duration on exception**:
  - HTTP `429` / `"rate limit"` / `"too many requests"`: Cooled for `self.cooldown_seconds` (default `20.0s`).
  - Any other error: Cooled for `min(5.0, self.cooldown_seconds)`.
- **Success clears cooldown**: The winning provider's cooldown is popped and `router.last_used_provider` is updated.
- **Telemetry**: Every switch calls `tracer.record_failover(from_label, to_label, reason)`, visible in `agent.get_last_run_metrics()["failover_events"]`.

> [!CAUTION]
> **Router & `fast_mode` Interaction**:
> - If an `LLMRouter` mixes a cloud `OpenAILLM` and a backup `LocalLLM`, `Agent._is_local_llm(router)` returns `False` (`fast_mode=False`). Pass `fast_mode=True` explicitly if you still want ReAct-by-default and `fast_reflect`.
> - In `generate_stream()`, failover occurs **only if the provider fails before emitting chunks**. Once a stream begins yielding chunks to the caller, mid-stream exceptions propagate directly.
> - **Don't forget `fallback_models` vs `LLMRouter`**: Use `OpenAILLM(fallback_models=[...])` when rotating models on the **same** API key & URL; use `LLMRouter` when rotating across **different keys, platforms, or Cloud + Local** backends.
