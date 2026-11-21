# 05 — 3-Tier Memory, Context Budgeting, Checkpointing & Custom Memory

Covers:
- [`phoenix/framework/agent/memory/manager.py`](../../phoenix/framework/agent/memory/manager.py) (`MemoryManager` / `ContextMemory` / `HybridMemory` / `HybridMemoryManager`)
- [`phoenix/framework/agent/memory/cell.py`](../../phoenix/framework/agent/memory/cell.py) (`MemoryCell`)
- [`phoenix/framework/agent/memory/short_term/stm_manager.py`](../../phoenix/framework/agent/memory/short_term/stm_manager.py) (`ShortTermMemoryManager`, `ShortMemoryCell`)
- [`phoenix/framework/agent/memory/long_term/ltm_manager.py`](../../phoenix/framework/agent/memory/long_term/ltm_manager.py) (`LongTermMemoryManager`, `LongMemoryCell`)
- [`phoenix/framework/agent/memory/session.py`](../../phoenix/framework/agent/memory/session.py) (`SessionMemory`)
- [`phoenix/framework/agent/memory/reflection.py`](../../phoenix/framework/agent/memory/reflection.py) (`ReflectionMemory`)
- [`phoenix/framework/agent/memory/budget.py`](../../phoenix/framework/agent/memory/budget.py) (`estimate_tokens`, `truncate_text`, `compact_value`, `fit_sections`)
- [`phoenix/framework/agent/memory/adapter.py`](../../phoenix/framework/agent/memory/adapter.py) (`InteractiveMemoryAdapter`)

---

## 1. The 3-Tier Memory Architecture

`MemoryManager` enforces hard bounds on both RAM growth and LLM prompt token consumption across three distinct tiers plus Session & Reflection stores:

```
  ┌─────────────────────────────────────────────────────────────────────┐
  │ MemoryManager(max_context_chars=8000, max_context_tokens=1800, ...) │
  │                                                                     │
  │  Tier 1: Working Memory (self.cells: Dict[str, MemoryCell])         │
  │    - Per-task execution slice (tool_history, io_history, errors)    │
  │    - Scoped to one run; cleared on set_session(); compacted on end  │
  │                                                                     │
  │  Tier 2: Short-Term Memory (ShortTermMemoryManager)                 │
  │    - Per-session sliding window (default 12 turns)                  │
  │    - Oldest evicted turns fold into zero-cost rolling summary       │
  │                                                                     │
  │  Tier 3: Long-Term Memory (LongTermMemoryManager)                   │
  │    - Bounded (default 200 items), SHA-1 deduplicated facts/lessons  │
  │    - Optional JSONL persistence via `ltm_storage_path`              │
  │                                                                     │
  │  Helpers: SessionMemory (<=40 vars) & ReflectionMemory (<=10 rules) │
  └─────────────────────────────────────────────────────────────────────┘
```

### 1.1 `MemoryManager.__init__` Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `max_context_chars` | `int` | `8000` | Character guardrail (clamped `>= 400`). |
| `max_context_tokens` | `int` | `1800` | Soft token cap for `get_full_context`, `get_task_context`, `get_execution_digest` (`min(max(100, max_context_tokens), max_context_chars // 4)`). |
| `stm_window` | `int` | `12` | Max recent messages retained verbatim per `session_id` before oldest messages roll into summary. |
| `ltm_max_items` | `int` | `200` | Maximum items retained in `LongTermMemoryManager` before lowest-value eviction. |
| `ltm_storage_path` | `Optional[str]` | `None` | Optional file path (`.jsonl`) to load/persist Long-Term Memory across restarts. |

---

## 2. Detailed Breakdown of Each Memory Tier

### 2.1 Tier 1 — Working Memory (`MemoryCell`)

Each planned task (or `"react_main"` in `react` mode) receives one `MemoryCell` dataclass instance:

- **Default Bounds**: `max_attempts=5` (overridden to `2` when `AgentLoop.fast_reflect=True`), `max_history=8` (io/tool/problem/solution/action lists), `max_tool_output_chars=2400`, max `4` reflections, `4` runtime results, `5` deduplicated `error_log` entries.
- **Deduplication**: `add_runtime_result()` intentionally stores execution counts & short responses rather than re-copying verbose tool outputs already stored in `tool_history`.
- **`cell.compact()`**: Called automatically at the end of a run (`finalize_run`) on `done`/`failed`/`skipped` cells to drop heavy history buffers while keeping compact digests intact.

### 2.2 Tier 2 — Short-Term Memory (`ShortTermMemoryManager`)

```python
ShortTermMemoryManager(max_cells=12, max_message_chars=1200, max_summary_chars=900)
```

- Every message stored via `await memory.add_interaction(session_id, role, content, metadata)` is middle-truncated to `max_message_chars` (preserving head & tail).
- When `len(session_bucket) >= max_cells`, the oldest `ShortMemoryCell` pops off and `_fold_evicted()` appends a 140-char snippet into the session's rolling `_summaries[session_id]` string (clamped at `900` chars) at **zero LLM token cost**.
- Optional LLM compression: `await memory.short_term.compress_session(session_id, llm)` summarizes older turns (`[:-4]`) into <= 4 bullet points using `llm.generate(..., max_tokens=200)`.
- **Filtering noise**: `MemoryManager.add_interaction()` automatically skips `role="system"` progress logs unless `metadata={"persist": True}`, and strips verbose `"\n\n---\nExecution Summary:\n"` footers before saving assistant turns!

### 2.3 Tier 3 — Long-Term Memory (`LongTermMemoryManager`)

```python
LongTermMemoryManager(max_items=200, max_item_chars=360, storage_path="./agent_ltm.jsonl")
```

- **Deduplication**: Normalized whitespace+lowercase SHA-1 fingerprint (`_fingerprint`). Adding a duplicate fact boosts existing `importance` (`+0.05`, capped at `1.0`) and increments `access_count` instead of creating duplicates.
- **Hybrid Retrieval Scoring (`retrieve(query="", limit=4, session_id=None, query_embedding=None, min_score=0.08)`)**:
  - Cross-session facts (`memory_type != "outcome"`) are shared globally; `"outcome"` memories remain session-scoped when `session_id` is passed.
  - Score = `0.60 * relevance + 0.25 * importance + 0.15 * recency`, where `relevance = max(token_overlap, cosine_similarity)` and `pinned=True` metadata guarantees minimum `0.35` relevance!
- **Automatic Eviction (`_evict_lowest_value`)**: When `len(_cells) >= max_items`, drops the lowest utility entry based on `importance * 0.7 + access_bonus - age_decay`.

### 2.4 `SessionMemory` & `ReflectionMemory`

- **`SessionMemory(max_vars=40)`**: Bounded FIFO key-value store (`set`, `get`, `get_all`, `to_context_string`, `clear`). Updated via `add_context_item(...)` or interaction metadata.
- **`ReflectionMemory(consolidation_threshold=5, max_reflections=10, max_item_chars=240)`**: Deduplicates critiques/lessons; when `finalize_run()` detects `len(reflections) >= consolidation_threshold`, it invokes `await reflection.consolidate(llm)` and promotes the distilled rules into Long-Term Memory as `memory_type="lesson"`.

---

## 3. Token Budgeting & Priority Packing (`budget.py`)

Implemented in [`phoenix/framework/agent/memory/budget.py`](../../phoenix/framework/agent/memory/budget.py):

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `estimate_tokens(text)` | `-> int` | Fast ~4 chars/token estimate (`(len(text) + 3) // 4`). |
| `truncate_text(text, max_chars, mode="tail_ellipsis")` | `-> str` | Modes: `'tail_ellipsis'` (clean line-aware head), `'middle'` (65% head + `\n... [truncated] ...\n` + 35% tail), `'tail'` (recent suffix). |
| `compact_value(value, max_chars=800)` | `-> str` | Strips null/empty dict fields; special compact extractors for `FileReadResult` (extracts `[L1-100]` code lines & symbol outline without verbose Pydantic fields) and `CommandExecutionResult` (`exit=.. \| stdout: ..`). |
| `fit_sections(sections, max_tokens)` | `-> str` | Packs `(priority, header, body)` tuples in priority order (`0` highest), greedily fitting whole sections and line-truncating lower-priority sections to fit `max_tokens`. |

### Key `MemoryManager` Context Methods

| Method | Signature | Priority Order & Purpose |
| :--- | :--- | :--- |
| `add_interaction` | `async (session_id, role, content, metadata=None)` | Appends conversational turn to STM & updates session vars from `metadata`. |
| `add_context_item` | `async (session_id, key, value, source="user", long_term=True, pinned=False)` | Sets `SessionMemory[key]` + optionally stores in LTM (`importance=0.85` if `pinned` else `0.75`). |
| `get_full_context` | `async (session_id="", query="", max_tokens=None) -> str` | Priority `0`: `OBJECTIVE`, `1`: `SESSION VARIABLES` & `CONVERSATION HISTORY`, `2`: `LONG-TERM MEMORY` (skipping keys already in session vars!) & `Reflections`, `3`: `COMPLETED TASKS`. |
| `get_context_bundle` | `async (session_id, query="") -> Dict[str, Any]` | Returns dict with `session_id`, `session_variables`, `global_context`, `full_text`, `stats`. |
| `get_task_context` | `(task_id, max_tokens=None) -> str` | Dependency-aware DAG slice: Priority `0`: Objective & `CURRENT TASK CONTEXT`, Priority `1`: `DEPENDENCY OUTPUTS` (direct upstream task outputs up to 800 chars each), Priority `2`: `SESSION STATE`, Priority `3`: 1-line summaries of unrelated completed tasks. |
| `build_retry_context` | `(task_id, max_tokens=None) -> str` | Tight `min(1200, max_context_tokens)` non-redundant slice for `Reflector` & `thinker.replan`. |
| `get_execution_digest` | `(max_tokens=None) -> str` | Compiles finished task findings, last 3 tool outcomes, and ratings for `synthesize_final_answer`. |
| `get_stats()` | `-> Dict[str, Any]` | Telemetry on working cells, STM active & evicted counts, LTM items, reflections, session vars, budget. |

---

## 4. Checkpointing & Crash Recovery

`MemoryManager` persists full runnable state atomically (`write to .tmp -> os.replace()`):

- **`memory.save_checkpoint(checkpoint_path, plan_data=None, prompt="") -> str`**
- **`await memory.load_checkpoint(checkpoint_path) -> Dict[str, Any]`**

Saved JSON payload contains:
`session_id`, `prompt`, `objective`, `plan`, `task_order`, `completed_tasks`, `failed_tasks`, `cells` (every `MemoryCell`), `session_vars`, `reflections`, `stm_summary`, and `stm_turns`.

When configured via `Agent(checkpoint_dir="./checkpoints")`, checkpoints are written automatically in `mode="plan"` and restored via `await agent.resume(session_id)`.

---

## 5. Integrating Custom Memory (`InteractiveMemoryAdapter`)

If you pass your own Redis/SQL/Vector memory class to `Agent(memory=custom_mem)`:
- If it already defines **both** `add_interaction` and `get_full_context`, `Agent` uses it directly.
- Otherwise `InteractiveMemoryAdapter` wraps it cleanly, translating:
  - Writes -> calls `add_interaction`, else `add_message`, else `store`.
  - Reads -> calls `get_full_context`, else `get_context`.
  - Reflections -> attaches `SessionMemory` and `ReflectionMemory` automatically if absent on your class.

---

## 6. Memory Cautions & Requirements

> [!CAUTION]
> 1. **Working Cells Reset Every Run**: Calling `agent.run(..., mode="plan")` or `mode="react"` calls `active_memory.set_session(session_id, ...)`, which clears `cells`, `task_order`, `completed_tasks`, and `failed_tasks` for the new run while **preserving** `short_term`, `long_term`, and `session` across runs!
> 2. **STM Is In-Memory Unless Checkpointed**: `ltm_storage_path` persists **only** `LongTermMemoryManager` items (facts, lessons, outcomes). Conversational STM turns (`ShortTermMemoryManager`) persist across process restarts **only** via `save_checkpoint` / `load_checkpoint` (in `plan` mode with `checkpoint_dir`) or a custom persistent `memory` object.
> 3. **Pinned Facts Bypass Query Overlap**: Setting `pinned=True` in `add_context_item(..., pinned=True)` ensures key domain rules remain retrievable in `get_full_context` even when user wording doesn't match keyword tokens.
> 4. **Multi-Turn Memory With `Agent.run`**: Passing `session_id=None` generates a **new random UUID every call**, isolating runs from each other. Always pass a consistent `session_id` (e.g. `session_id="user_42_erp"`) for multi-turn conversations!
