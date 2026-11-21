# 04 — Creating Tools, Registry, Hooks & Runtime Sandbox

Covers:
- [`phoenix/framework/agent/tools/base.py`](../../phoenix/framework/agent/tools/base.py) (`BaseTool`, `FunctionTool`, `CustomTool`, `ToolResult`, `@tool`, `create_tool`, `ensure_tool`)
- [`phoenix/framework/agent/tools/registry.py`](../../phoenix/framework/agent/tools/registry.py) (`ToolRegistry`)
- [`phoenix/framework/agent/execution/tool_manager.py`](../../phoenix/framework/agent/execution/tool_manager.py) (`ToolManager`)
- [`phoenix/framework/agent/tools/`](../../phoenix/framework/agent/tools/) & [`bank/`](../../phoenix/framework/agent/tools/bank/) (12 Default Tools + 8 Tool Bank Integrations)
- [`phoenix/framework/agent/runtime/restricted_python.py`](../../phoenix/framework/agent/runtime/restricted_python.py) (`RestrictedPythonRuntime`)

---

## 1. All Ways to Create Custom Tools

### 1.1 `@tool` Decorator (4 Calling Styles)

```python
from typing import Annotated, Literal, Optional
from pydantic import BaseModel, Field
from phoenix.framework.agent import tool

# Style 1: Bare decorator (name = function name, description = docstring)
@tool
def get_stock(sku: str) -> dict:
    """Return available stock and warehouse location for a SKU.

    Args:
        sku: Product SKU code such as LAPTOP-PRO.
    """
    return {"sku": sku, "stock": 3}

# Style 2: Explicit name only (description from docstring)
@tool("lookup_customer")
async def get_customer(customer_id: str) -> dict:
    """Fetch customer balance and credit limit asynchronously."""
    return {"id": customer_id, "credit_limit": 4000}

# Style 3: Positional name & description
@tool("calc_tax", "Compute VAT amount for an order subtotal")
def tax_calc(subtotal: float, rate: float = 0.14) -> float:
    return round(subtotal * rate, 2)

# Style 4: Keyword configuration with Pydantic args_schema, read_only, return_direct
class InvoiceInput(BaseModel):
    customer_id: str = Field(..., description="Customer ID (e.g. C1)")
    sku: str = Field(..., description="Inventory SKU")
    qty: int = Field(..., ge=1, description="Units ordered")

@tool(
    name="create_order_invoice",
    description="Validate stock and credit limit, then issue an invoice.",
    args_schema=InvoiceInput,
    read_only=False,
    return_direct=False,
)
def issue_invoice(customer_id: str, sku: str, qty: int) -> dict:
    return {"invoice_id": "INV-105", "customer_id": customer_id, "sku": sku, "qty": qty}
```

### 1.2 `@agent.tool` Bound Decorator

Defines a tool and immediately registers it into that specific `Agent` (also auto-allowing it in `agent.profile.tool_access` if an allowlist exists):

```python
@agent.tool("check_credit", read_only=True)
def check_credit(customer_id: str) -> dict:
    """Inspect remaining credit for a customer."""
    return {"customer_id": customer_id, "available": 2800}
```

### 1.3 Subclassing `BaseTool`

Use `BaseTool` when your tool holds state (DB pool, API client, workspace directory) or needs custom initialization:

```python
from phoenix.framework.agent import BaseTool, ToolResult

class WarehouseLookupTool(BaseTool):
    name = "warehouse_lookup"
    description = "Query real-time stock from the ERP warehouse service."
    read_only = True
    return_direct = False

    def __init__(self, dsn: str, **kwargs):
        super().__init__(**kwargs)
        self.dsn = dsn

    async def execute(self, sku: str, warehouse: Optional[str] = None) -> ToolResult:
        return ToolResult(
            success=True,
            output={"sku": sku, "warehouse": warehouse or "Cairo-HQ", "qty": 12},
        )
```

> [!TIP]
> **Sync or Async**: Both `@tool` functions and `BaseTool.execute` can be synchronous `def` or `async def`. Synchronous functions are automatically offloaded via `asyncio.to_thread(...)` inside `invoke()` / `FunctionTool.execute()`, and `@tool`-decorated functions remain directly callable in normal Python (`tax_calc(100)` works!).

### 1.4 Programmatic Helpers (`FunctionTool`, `create_tool`, `ensure_tool`)

- `FunctionTool(func, name=None, description=None, args_schema=None, read_only=False, return_direct=False, runtime=None)` (`CustomTool` is an alias). Also supports `FunctionTool.from_function(...)`.
- `create_tool(func, name=None, description=None, args_schema=None, read_only=False, return_direct=False) -> FunctionTool`.
- `ensure_tool(item, name=None, description=None, **kwargs) -> BaseTool` accepts:
  1. A `BaseTool` instance (`FunctionTool`, custom subclass)
  2. A `BaseTool` subclass (auto-instantiated with zero args)
  3. Any Python callable (sync or async)
  4. A dict: `{"func": my_fn, "name": "my_tool", "description": "...", "read_only": True}`

So all of these are valid in `Agent(tools=[...])`:
```python
Agent(llm=llm, tools=[
    get_stock,                       # @tool / FunctionTool
    WarehouseLookupTool(dsn="..."),  # BaseTool instance
    MyZeroArgToolClass,              # BaseTool subclass
    plain_python_fn,                 # Plain function
    {"func": fn2, "name": "f2", "description": "Helper"},
])
```

---

## 2. Automatic JSON Schema Extraction, Coercion & Result Normalization

Implemented in [`phoenix/framework/agent/tools/base.py`](../../phoenix/framework/agent/tools/base.py):

### 2.1 Parameter Type Mapping (`_python_type_to_json_schema`)

| Python Annotation | JSON Schema Produced |
| :--- | :--- |
| `str` / untyped / `Any` | `{"type": "string"}` |
| `int` | `{"type": "integer"}` |
| `float` | `{"type": "number"}` |
| `bool` | `{"type": "boolean"}` |
| `list`, `tuple`, `set` / `List[T]` | `{"type": "array", "items": ...}` |
| `dict` / `Dict[K, V]` | `{"type": "object"}` |
| `Optional[T]` / `Union[T, None]` | Unwraps `T` |
| `Literal["a", "b"]` / `Enum` | `{"type": ..., "enum": [...]}` |
| `Annotated[int, "desc"]` / `Annotated[int, Field(description="...")]` | Adds `"description"` directly onto property schema |
| Pydantic `BaseModel` subclass | Full `model_json_schema()` |

- Parameters without a default value (excluding `self`, `cls`, `*args`, `**kwargs`) are added to `"required"`. Non-`None` defaults become `"default"`.
- Docstrings are parsed for **Google style** (`Args:\n    param (type): desc`) and **Sphinx/reST style** (`:param name: desc`).
- If `args_schema` (a Pydantic `BaseModel` class) is passed, its schema overrides signature inspection cleanly and validates incoming arguments before execution!

### 2.2 Automatic Argument Coercion & Output Normalization

- **Coercion (`_coerce_value`)**: Converts stringified LLM outputs cleanly when safe — e.g. `"42" -> 42` for `int`, `"3.5" -> 3.5` for `float`, `"true"/"1"/"yes"` -> `True` / `"false"/"0"/"no"` -> `False` for `bool`, and `dict -> BaseModel`!
- **Unexpected Kwarg Protection (`_prepare_call_kwargs`)**: Unless the target function accepts `**kwargs`, extra keys hallucinated by an LLM are silently filtered out instead of crashing with `TypeError: got an unexpected keyword argument`.
- **`ToolResult` & `_normalize_tool_output`**:
  - Any `ToolResult(success=bool, output=Any, error=Optional[str])` is preserved as-is.
  - A dict containing `"success"` and (`"output"` or `"error"`) is normalized into `ToolResult`.
  - A Pydantic `BaseModel` return value is dumped to `dict` and wrapped in `ToolResult(success=True, output=dumped)`.
  - Any other return value (`dict`, `list`, `str`, `int`, etc.) becomes `ToolResult(success=True, output=result)`.
  - Uncaught exceptions inside `invoke()` / `FunctionTool.execute()` are caught and returned as `ToolResult(success=False, error=str(e))` so one tool error lets the agent recover instead of crashing the whole run!

### 2.3 Key `BaseTool` Methods

| Method | Signature | Description |
| :--- | :--- | :--- |
| `execute(**kwargs)` | `async -> ToolResult \| Any` | Main implementation hook (override in subclasses). |
| `invoke(**kwargs)` | `async -> ToolResult` | Safe entrypoint: validates/coerces kwargs, handles sync/async, normalizes `ToolResult`. |
| `run(**kwargs)` | `async -> Any` | Convenience helper returning `result.output` on success else `result.error`. |
| `get_parameters_schema()` | `-> Dict[str, Any]` | Returns `{"type": "object", "properties": ..., "required": ...}`. |
| `to_dict()` / `schema()` | `-> Dict[str, Any]` | Compact tool descriptor (`name`, `description`, `parameters`, optional `read_only`/`return_direct`). |
| `to_openai_tool()` | `-> Dict[str, Any]` | OpenAI Chat Completions `{"type": "function", "function": {...}}` payload. |

---

## 3. `ToolRegistry` & `ToolManager` (Permissions & Hooks)

### 3.1 `ToolRegistry` ([`registry.py`](../../phoenix/framework/agent/tools/registry.py))

| Method | Signature | Description |
| :--- | :--- | :--- |
| `__init__` | `(runtime=None, tools=None)` | Creates registry and optionally batch-registers `tools`. |
| `register` | `(tool_or_func, name=None, description=None, **kwargs) -> BaseTool` | Normalizes via `ensure_tool`, injects `self.runtime` if tool runtime is `None`, stores under `tool.name`. |
| `register_many` | `(tools: Iterable[Any]) -> List[BaseTool]` | Batch registers tools. |
| `register_function` | `(func, name=None, description=None, **kwargs) -> BaseTool` | Explicit `create_tool` + `register`. |
| `unregister` | `(name: str) -> bool` | Deletes tool if present. |
| `set_runtime` | `(runtime: Any)` | Rebinds runtime across registry and all registered tools. |
| `get_tool` / `has_tool` / `list_tools` | Lookup & listing methods | `get_tool` raises `ValueError` if missing. |
| `get_all_tools_info()` / `to_openai_tools()` | Schema exporters | Used by `ToolManager` and `Thinker`. |
| `ToolRegistry.load_default(runtime=None, extra_tools=None)` | `@classmethod` | Builds a registry containing all **12 default system tools** + `extra_tools`. |

### 3.2 `ToolManager` & Execution Hooks ([`tool_manager.py`](../../phoenix/framework/agent/execution/tool_manager.py))

```python
ToolManager(
    registry: ToolRegistry,
    profile: Optional[AgentProfile] = None,
    runtime: Optional[BaseRuntime] = None,
    before_tool_hooks: Optional[List[Callable]] = None,
    after_tool_hooks: Optional[List[Callable]] = None,
)
```

`ToolManager` gates profile permissions (`is_tool_allowed`), exports permitted schemas (`get_available_tools_info()`, `get_openai_tools()`), records `tracer.record_tool(...)` latency/failures, and runs **sync or async hooks**:

- **Before Hooks** (`agent.tool_manager.add_before_hook(fn)` or `register_before_hook(fn)`):
  - Signature: `fn(tool_name: str, payload: Dict[str, Any]) -> Optional[Union[bool, ToolResult]]`
  - **Mutate arguments**: Modify `payload` dict in place before execution.
  - **Block execution**: Return `False` -> returns `ToolResult(success=False, error="Execution of tool '...' was rejected by a before-tool hook.")`.
  - **Mock / short-circuit**: Return a `ToolResult(...)` -> skips calling the actual tool and returns that result.
- **After Hooks** (`agent.tool_manager.add_after_hook(fn)` or `register_after_hook(fn)`):
  - Signature: `fn(tool_name: str, payload: Dict[str, Any], result: ToolResult) -> Optional[ToolResult]`
  - Inspect `result` for logging/auditing, or return a replacement `ToolResult` (e.g. redacting secrets!).

---

## 4. Built-In Tools Reference

### 4.1 The 12 Default Tools (`ToolRegistry.load_default()`)

Loaded automatically ONLY when `Agent(tools=None)` (default) OR `Agent(..., include_default_tools=True)`:

| Tool Name | Class | Key Input Arguments | Read-Only (`Actor`) | Notes / Requirements |
| :--- | :--- | :--- | :--- | :--- |
| `web_search` | `WebSearchTool` | `query: str`, `max_results: int = 3` | Yes | Requires `duckduckgo-search` (`[agent]`) and internet access |
| `python_repl` | `CodeExecutionTool` | `code: str` | No | Runs inside `RestrictedPythonRuntime.execute_code` |
| `python_analyzer` | `PythonAnalyzerTool` | `file_path: str` | Yes | Pure AST index of classes, methods, functions, imports with line numbers |
| `execute_command` | `CommandExecutionTool` | `command: str`, `cwd: Optional[str] = None` | No | Shell execution via `RestrictedPythonRuntime.execute_command` |
| `compile_code` | `CodeCompileTool` | `file_path: str` | No | Runs `py_compile.compile(..., doraise=True)` |
| `file_read` | `FileReadTool` | `file_path: str`, `from_line=1`, `to_line`, `block_size=100` | Yes | Reads file chunks or directory outline; auto-corrects paths |
| `file_write` | `FileWriteTool` | `file_path: str`, `content: str` | No | Writes/overwrites files via runtime IO |
| `file_append` | `FileAppendTool` | `file_path: str`, `content: str` | No | Appends content to file |
| `file_edit` | `FileEditTool` | `file_path`, line/chunk params | No | Edits file content |
| `file_search` | `FileSearchTool` | `file_path: str`, `search_query: str`, `is_regex=False`, `case_sensitive=False` | Yes | Regex or substring match inside file |
| `file_delete` | `FileDeleteTool` | `file_path: str` | No | Deletes file via runtime IO |
| `file_update_multi` | `MultiBlockUpdateTool` | `file_path: str`, `edits: List[{target, replacement}]` | No | Replaces multiple unique exact blocks atomically |

### 4.2 The 8 Tool Bank Extensions (`phoenix.framework.agent.tools.bank`)

Opt-in specialized integrations imported explicitly and passed into `Agent(tools=[...])`:

```python
from phoenix.framework.agent.tools.bank import (
    WebScraperTool, WikipediaSearchTool, ArxivSearchTool,
    GitHubTool, SlackTool, EmailTool,
    SQLDatabaseTool, APIRESTTool,
)
```

| Tool Name | Class | Required / Key Parameters | Env Vars & Requirements |
| :--- | :--- | :--- | :--- |
| `sql_database` | `SQLDatabaseTool(connection_uri=None, read_only=True)` | `action: "get_schema" \| "execute_query"`, `query?`, `connection_uri?` | `SQL_CONNECTION_URI`, `sqlalchemy`. **Caution:** defaults to `read_only=True` (allows only `SELECT`, `PRAGMA`, `EXPLAIN`). |
| `api_rest` | `APIRESTTool()` | `method`, `url`, `headers?`, `json_body?`, `params?` | Core `httpx`; output capped at `15,000` chars |
| `web_scraper` | `WebScraperTool()` | `url: str` | `requests` + `beautifulsoup4`; strips nav/script/footer, caps `15,000` chars |
| `wikipedia_search` | `WikipediaSearchTool()` | `query: str` | `requests` |
| `arxiv_search` | `ArxivSearchTool()` | `query: str`, `max_results: int = 3` | `requests` |
| `github` | `GitHubTool()` | `action: "search_repos" \| "get_file" \| "get_issue" \| "list_commits"`, `repo?`, `path?`, `issue_number?`, `query?`, `github_token?` | `GITHUB_TOKEN` (optional for public repos) |
| `slack` | `SlackTool()` | `message: str`, `webhook_url?` | `SLACK_WEBHOOK_URL` |
| `email` | `EmailTool()` | `recipient`, `subject`, `body`, `smtp_server?`, `smtp_port?`, `sender_email?`, `sender_password?` | `SMTP_SERVER` (`smtp.gmail.com`), `SMTP_PORT` (`587`), `SMTP_EMAIL`, `SMTP_PASSWORD` |

---

## 5. `RestrictedPythonRuntime` & Sandbox Cautions

[`phoenix/framework/agent/runtime/restricted_python.py`](../../phoenix/framework/agent/runtime/restricted_python.py):

```python
runtime = RestrictedPythonRuntime(
    workspace_root="/path/to/allowed/project",
    timeout=30.0,
)
agent = Agent(llm=llm, runtime=runtime, include_default_tools=True)
```

### Security Rules Enforced

1. **AST Code Inspection (`execute_code`)**:
   - **Blocked modules**: `os`, `sys`, `subprocess`, `shutil`, `pty`, `socket`, `builtins`, `macpath`, `ntpath`, `posixpath`, `cmd`, `shlex`, `importlib`, `ctypes`, `pickle`, `marshal`, `pathlib`, `tempfile`, `glob`.
   - **Allowed imports ONLY**: `math`, `json`, `re`, `datetime`, `collections`, `itertools`, `functools`, `statistics`, `random`, `decimal`, `fractions`, `hashlib`, `base64`, `uuid`, `string`, `typing`, `dataclasses`, `enum`, `copy`, `bisect`, `heapq`, `time`, `operator`.
   - **Blocked functions**: `eval`, `exec`, `open`, `compile`, `__import__`, `globals`, `locals`, `vars`, `breakpoint`, `input`.
   - **Blocked dunder attributes**: `__subclasses__`, `__bases__`, `__mro__`, `__globals__`, `__builtins__`, `__import__`, `__code__`, `__closure__`.
2. **Command Execution (`execute_command`)**:
   - Rejects destructive substrings (`rm -rf /`, `mkfs`, `:(){ :|:& };:`, `dd if=/dev/zero`, `> /dev/sda`, `chmod -R 777 /`).
   - Enforces `timeout` (default `30.0s`) and jails `cwd` inside `workspace_root` when configured.

> [!CAUTION]
> 1. **`execute_command` Is NOT a Container**: `execute_command` filters specific destructive strings and jails `cwd`, **not** arbitrary shell syntax or OS permissions. In multi-tenant or untrusted environments, disable `include_default_tools`, enforce `AgentProfile.tool_access`, or run inside an isolated container/VM.
> 2. **Tool Name Collisions**: Registering two tools with the same `name` overwrites the earlier entry in `ToolRegistry.tools`. Always give distinct `@tool("unique_name")` identifiers when registering multiple tools.
> 3. **Tool Output Size**: `MemoryCell` truncates stored `tool_history` outputs at `2400` chars (`compact_value`), and `run_react` compacts observations to `900` chars. Keep custom tool outputs concise dicts/JSON rather than huge raw dumps!
