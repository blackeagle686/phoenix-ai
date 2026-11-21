# 03 — Execution Loops (`AgentLoop`) & Cognition Architecture

Covers:
- [`phoenix/framework/agent/core/loop.py`](../../phoenix/framework/agent/core/loop.py) (`AgentLoop`)
- [`phoenix/framework/agent/cognition/schema.py`](../../phoenix/framework/agent/cognition/schema.py)
- [`phoenix/framework/agent/cognition/thinker/thinker.py`](../../phoenix/framework/agent/cognition/thinker/thinker.py) (`Thinker`)
- [`phoenix/framework/agent/cognition/planner/planner.py`](../../phoenix/framework/agent/cognition/planner/planner.py) (`Planner`)
- [`phoenix/framework/agent/cognition/actor/actor.py`](../../phoenix/framework/agent/cognition/actor/actor.py) (`Actor`)
- [`phoenix/framework/agent/cognition/reflector/reflector.py`](../../phoenix/framework/agent/cognition/reflector/reflector.py) (`Reflector`)
- [`phoenix/framework/agent/cognition/analyzer/analyzer.py`](../../phoenix/framework/agent/cognition/analyzer/analyzer.py) (`Analyzer`)
- [`phoenix/framework/agent/cognition/pipeline.py`](../../phoenix/framework/agent/cognition/pipeline.py) (`CognitionPipeline`, `BrainRegistry`)

---

## 1. The 4 Execution Modes Explained

When you call `await agent.run(prompt, session_id=..., max_iterations=15, mode=...)`, execution flows into one of 4 concrete paths:

| Mode | LLM Calls (Typical) | Tools Used? | Best For | Key Guardrails |
| :--- | :--- | :--- | :--- | :--- |
| **`"fast_ans"`** (aliases: `"fast_mode"`, `"fast"`, `"direct"`, `"chat"`) | **1** | No | Greetings, conceptual Q&A, pure summarization | Uses `memory.get_full_context()`, saves reply to STM |
| **`"react"`** | **1 – 2** (up to `max_iterations`) | Yes | Single-goal tool queries, ERP/DB lookups, local LLMs (`3B`–`7B`), chatbots | Duplicate tool-call signature guard; sliding 8-observation window; `return_direct` immediate stop |
| **`"plan"`** (with modern `Planner.generate_initial_plan`) | **3+** per task (`Plan` + `Solve` + `Reflect` + `Synthesize`) | Yes (parallel DAG) | Complex multi-step workflows, code generation + tests, dependency graphs, checkpointed long runs | Dependency gating; `max_concurrency` semaphore; adaptive replanning; atomic disk checkpoints |
| **`"plan"`** (with custom legacy `Planner.plan` only) | `1 + 3*iters` | Optional | Backward compatibility with custom action-batch planners returning `{"actions": [...]}` | Stops cleanly only when `has_executed_real_action` and (`is_finish` or `is_complete`) |

---

## 2. `AgentLoop` Complete Reference

### 2.1 `AgentLoop.__init__` Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `thinker` | `Thinker` | *(required)* | Reasoning engine (`solve_task`, `generate_plan`, `generate_reflection`, `replan`, `synthesize_final_answer`). |
| `planner` | `Planner` | *(required)* | Generates `PlanSchema` via `generate_initial_plan` or action dicts via `plan`. |
| `actor` | `Actor` | *(required)* | Executes tool calls via `ToolManager` (or runtime IO/code). |
| `reflector` | `Reflector` | *(required)* | Evaluates runtime outputs (`ReflectionSchema`). |
| `analyzer` | `Optional[Analyzer]` | `None` | Scans workspace files before planning when `enable_workspace_analysis=True`. |
| `memory_manager` | `Optional[MemoryManager]` | `None` | Creates a fresh `MemoryManager()` if `None` or non-`MemoryManager`. |
| `parallel_tasks` | `bool` | `True` | Executes dependency-ready tasks concurrently via `asyncio.gather` in `plan` mode. |
| `max_concurrency` | `int` | `3` | `asyncio.Semaphore` limit for parallel ready tasks (clamped `>= 1`). |
| `replan_on_failure` | `bool` | `True` | Calls `thinker.replan()` if a task fails or `reflection.needs_replan == True`. |
| `synthesize_answer` | `bool` | `True` | Calls `thinker.synthesize_final_answer()` over `memory.get_execution_digest()` and appends `\n\n---\nExecution Summary:\n...`. |
| `enable_workspace_analysis` | `bool` | `False` | Calls `analyzer.analyze_workspace(prompt)` before generating `PlanSchema`. |
| `approval_callback` | `Optional[Callable[[TaskSchema, str], Any]]` | `None` | Sync or async HITL gate called before each `react` tool step or `plan` task execution. Return `"skip"` to skip, `"reject"` / `False` to abort cleanly, or `"approve"` / `True` / `None` to proceed. |
| `checkpoint_dir` | `Optional[str]` | `None` | Directory for `<safe_session_id>.json` atomic checkpoints saved after plan creation & every DAG wave. |
| `fast_reflect` | `bool` | `False` | When `True` (set automatically when `agent.fast_mode=True`): caps task `max_attempts` from `5` to `2`, limits `max_replans` from `3` to `1`, and skips the LLM `Reflector` call whenever a task succeeded with >=1 successful tool/direct finding and zero failures. |

---

### 2.2 Inside `mode="react"` (`run_react` & `run_react_stream`)

`run_react` eliminates planning, reflection, and multi-pass synthesis overhead — making it ideal for local models and low-latency tool agents:

1. **Initialization**: Fetches `base_context = await memory.get_full_context(session_id, query=prompt)`, resets working memory via `active_memory.set_session(session_id, prompt)`, creates a single `MemoryCell("react_main")`, and emits `"react_start"`.
2. **Loop (`step = 1..max_iterations`)**:
   - Builds compact `step_context` with `base_context` + last **8** tool observations (`observations[-8:]`) + anti-repeat instruction.
   - Calls `task_exec = await thinker.solve_task(react_task, context=step_context)`.
   - **No tools**: Returns `direct_response` (or `thought_process` or `observations`), marks `"react_main"` complete, breaks.
   - **Duplicate Tool-Call Loop Guard (`step > 1`)**: Computes signature `(tool_name, compact_value(arguments, max_chars=300))` for every call. If **all** tool calls in the current step already executed identically in earlier steps: breaks out of the loop and synthesizes a final answer (or returns `direct_response` / `observations`) — **stopping infinite small-LLM retry loops**!
   - **Human Approval Gate**: If `approval_callback` is set, awaits decision before executing tools.
   - **Tool Execution**: Executes each `ToolCall` sequentially via `actor._invoke_tool()`, logs to `cell.add_tool_record()`, compacts output/error to `900` chars (`[Step {step} | {t_name} | {status}]: ...`).
   - **`return_direct=True` Short-Circuit**: If any successful tool has `return_direct=True`, sets `final_answer = str(rec["output"])` and exits immediately without another LLM turn.
   - **Early Complete**: If `task_exec.is_complete` and `direct_response` is non-empty, exits immediately.
3. **Finalization**: Saves assistant message to STM, schedules `active_memory.finalize_run(...)`, flushes background tasks, emits `"complete"`.

> [!CAUTION]
> **`max_iterations` Default in `run_react`**: Calling `agent.run(..., mode="react")` uses `Agent.run`'s default `max_iterations=15`, whereas calling `loop.run_react(...)` directly defaults to `max_iterations=6`. Always pass `max_iterations=6` (or similar bound) for interactive chat UIs.

---

### 2.3 Inside `mode="plan"` (`_run_plan_dag_execution`)

1. Optional `workspace_analysis` via `Analyzer.analyze_workspace(prompt)`.
2. `plan: PlanSchema = await planner.generate_initial_plan(...)` creates ordered `TaskSchema` objects (`task_id`, `dependencies`, `task_type`, `priority`).
3. Creates a `MemoryCell` per task (`active_memory.create_cell(...)`) and saves initial checkpoint.
4. **DAG Wave Execution Loop** (`while remaining and iterations < max_iterations`):
   - Selects `ready` tasks whose `dependencies` are all in `completed_tasks` (fallback: `[remaining[0]]` if cyclic/unmet dependencies).
   - **Parallel Execution**: Runs up to `max_concurrency` ready tasks simultaneously iff `parallel_tasks=True`, `len(ready) > 1`, **and `approval_callback is None`** (disables parallelism automatically during HITL!).
   - Inside `_execute_single_task`:
     - Checks `approval_callback`.
     - Calls `actor.generate_and_execute(task, context=active_memory.get_task_context(task.task_id))` — note `Actor` itself can perform **up to `max_tool_steps=2` inner sub-steps** if Step 1 only called `read_only` tools on a write/update/incomplete task!
     - Records `io_history`, `tool_history`, `runtime_results`, `error_log`.
     - Evaluates via `fast_reflect` heuristic OR `Reflector.reflect(ReflectorInputSchema(...))`.
   - **Replanning**: On failure or `reflection.needs_replan`, up to `max_replans` (`1` if `fast_reflect` else `3`), calls `thinker.replan(...)` and appends new non-duplicate `TaskSchema` nodes to the graph.
   - Writes checkpoint after every wave.
5. **Final Synthesis & Cleanup**: Calls `thinker.synthesize_final_answer(...)`, stores interaction in STM, promotes outcomes/reflections to LTM (`finalize_run`), compacts finished `MemoryCell`s, flushes async background tasks, emits `"complete"`.

---

## 3. Core Cognition Brain Modules & Schemas

### 3.1 Structured Pydantic Schemas (`cognition/schema.py`)

| Schema | Key Fields | Purpose |
| :--- | :--- | :--- |
| `ToolCall` (`ToolCallSchema`) | `tool_name: str`, `arguments: Dict[str, Any]` | Single tool invocation emitted by LLM/Thinker. |
| `TaskExecutionSchema` | `thought_process: str`, `tools_to_call: List[ToolCall]`, `direct_response: Optional[str]`, `is_complete: bool` | Output of `thinker.solve_task()` / `llm.generate_with_tools()`. |
| `TaskSchema` | `task_id: str`, `task_type: TaskType`, `priority: TaskPriority`, `description: str`, `status: TaskStatus`, `dependencies: List[str]` | Task node inside `PlanSchema`. |
| `PlanSchema` | `objective: str`, `tasks: List[TaskSchema]` | Output of `planner.generate_initial_plan()`. |
| `ReflectionSchema` | `status: TaskStatus`, `feedback: str`, `rating: int (1..10)`, `is_task_complete: bool`, `summary: Optional[str]`, `needs_replan: bool`, `confidence: float`, `reasoning: str` | Output of `reflector.reflect()`. |
| `ActorOutputSchema` | `task_id: str`, `tool_name: str`, `success: bool`, `result: Optional[Dict]`, `error_context: Optional[str]`, `reflection: Optional[Any]` | Normalized output of `actor.generate_and_execute()`. |

Enums:
- `TaskStatus`: `"pending"`, `"in_progress"`, `"done"`, `"failed"`, `"skipped"`, `"cancelled"`
- `TaskType`: `"read"`, `"write"`, `"search"`, `"update"`, `"delete"`, `"other"`
- `TaskPriority`: `"critical"`, `"high"`, `"medium"`, `"low"`

### 3.2 `Thinker` (`cognition/thinker/thinker.py`)

- `Thinker(llm, profile=None, tool_manager=None)`
- Key methods:
  - `analyze(prompt, memory=None, session_id="") -> str`
  - `generate_plan(prompt, memory=None, session_id="", workspace_analysis=None) -> PlanSchema`
  - `solve_task(task, context="") -> TaskExecutionSchema` — Automatically uses native OpenAI/Ollama tool calling (`llm.generate_with_tools`) when the LLM class overrides `BaseLLM.generate_with_tools` **and** `tool_manager.get_openai_tools()` is non-empty; otherwise injects full JSON tool definitions and calls `generate_structured(..., TaskExecutionSchema)`.
  - `generate_reflection(runtime_output, context="") -> ReflectionSchema`
  - `replan(objective, completed_summary, failed_task, failure_context) -> PlanSchema`
  - `synthesize_final_answer(prompt, objective, execution_digest, context="") -> str`

### 3.3 `Actor` (`cognition/actor/actor.py`)

- `Actor(tool_manager, thinker=None, runtime=None, reflector=None)`
- `generate_and_execute(task, previous_results="", context="", max_tool_steps=2) -> ActorOutputSchema`
- **Read-Only Follow-Up Rule (`_needs_followup_action_step`)**:
  Reads the builtin `READ_ONLY_TOOLS` set (`file_read`, `file_search`, `python_analyzer`, `web_search`, `wikipedia_search`, `arxiv_search`, `web_scraper`) **or** custom tools flagged `read_only=True`. If Step 1 only ran read-only tools and either `is_complete=False` OR the task asks to `write`/`create`/`edit`/`update`/`modify`/`patch`/`delete`/`append`/`implement`/`fix`, `Actor` injects Step 1 results and runs Step 2 so the agent reads first and writes second inside the same task!

### 3.4 `Analyzer` (`cognition/analyzer/analyzer.py`)

- `await analyzer.analyze_workspace(prompt, root_dir=".") -> Dict[str, Any]`
- Walks `root_dir` (excluding `.git`, `__pycache__`, `venv`, `.venv`, `.env`, `node_modules`, `.gemini`, `agent_venv`, `git_venv`, and any `*_venv`), caches up to `200` filenames, sends up to `100` to LLM (`max_tokens=150`), and returns `{"relevant_files": [...], "tech_stack": ..., "summary": ...}`.

### 3.5 Custom Components & JSON Cognition Pipelines

You can customize the loop at three levels:

1. **`component_factories` or `set_component()`**:
   ```python
   agent = Agent(
       llm=llm,
       component_factories={
           "loop": lambda thinker, planner, actor, reflector, analyzer, memory: AgentLoop(
               thinker=thinker,
               planner=planner,
               actor=actor,
               reflector=reflector,
               analyzer=analyzer,
               memory_manager=memory,
               parallel_tasks=False,
               enable_workspace_analysis=True,
               approval_callback=my_hitl_gate,
           )
       },
   )
   ```
   Or at runtime: `agent.set_component("thinker", my_thinker)` (automatically rebuilds `self.loop`).

2. **Custom `loop_cls` Subclass**:
   ```python
   agent = Agent(llm=llm, loop_cls=MyCustomAgentLoop)
   ```

3. **JSON Bootstrap Pipelines (`CognitionPipeline`)**:
   ```python
   # my_bootstrap.json
   # {"name": "custom_boot", "version": "1.0", "steps": [
   #   {"id": "s1", "brain": "analyzer", "input_map": {"prompt": "prompt"}, "output_key": "analysis", "required_inputs": ["prompt"]},
   #   {"id": "s2", "brain": "thinker",  "input_map": {"prompt": "prompt", "memory": "memory", "session_id": "session_id"}, "output_key": "thinker_output", "required_inputs": ["prompt"]}
   # ]}
   agent.set_cognition_pipeline(bootstrap_pipeline_path="my_bootstrap.json")
   ```
   *(Applies when using the iterative action-batch loop path `_run_iterative_action_loop`.)*

---

## 4. Loop Cautions & Common Pitfalls

> [!CAUTION]
> 1. **`Actor` Path Hint**: Both `Thinker.generate_plan` and `Thinker.solve_task` instruct the LLM to prefer **absolute paths** and native file tools (`file_read`, `file_write`, `file_edit`) over shell commands. Keep that in mind when designing coding prompts.
> 2. **HITL Disables Parallelism**: Setting `approval_callback=...` forces sequential execution (`can_parallelize = False`) in `plan` mode so multiple concurrent approval prompts do not collide.
> 3. **`return_direct=True` Skips Synthesis**: If a tool sets `return_direct=True`, `run_react` returns its raw `output` string immediately upon success without a second LLM call. Use this when returning large structured JSON tables or pre-formatted responses directly to a backend API!
> 4. **Max Iterations Exhausted**: In `plan` mode, if `iterations >= max_iterations` before all `remaining` tasks finish, the loop appends `"\nMax iterations reached before completing all tasks."` and still runs final synthesis & checkpoint saving so you can inspect progress or resume later.
