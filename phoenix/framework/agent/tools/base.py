import asyncio
import inspect
import re
import types
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, Union, get_args, get_origin, get_type_hints
from pydantic import BaseModel


class ToolResult(BaseModel):
    success: bool
    output: Any
    error: Optional[str] = None


def _resolve_json_refs(node: Any, defs: Dict[str, Any], seen: Optional[set] = None) -> Any:
    """Recursively inline Pydantic $defs/$ref references so OpenAI/LLM schemas stay self-contained."""
    seen = seen or set()
    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/"):
            ref_name = ref.split("/")[-1]
            if ref_name in defs and ref_name not in seen:
                resolved = _resolve_json_refs(defs[ref_name], defs, seen | {ref_name})
                merged = dict(resolved) if isinstance(resolved, dict) else {}
                for k, v in node.items():
                    if k != "$ref":
                        merged[k] = _resolve_json_refs(v, defs, seen)
                return merged
        return {
            k: _resolve_json_refs(v, defs, seen)
            for k, v in node.items()
            if k not in ("$defs", "definitions", "title")
        }
    if isinstance(node, list):
        return [_resolve_json_refs(item, defs, seen) for item in node]
    return node


_PRIMITIVE_ANNOTATION_NAMES: Dict[str, Any] = {
    "int": int,
    "float": float,
    "bool": bool,
    "str": str,
    "list": list,
    "dict": dict,
    "tuple": tuple,
    "set": set,
    "builtins.int": int,
    "builtins.float": float,
    "builtins.bool": bool,
    "builtins.str": str,
}


def _is_union_origin(origin: Any) -> bool:
    union_type = getattr(types, "UnionType", None)
    return origin is Union or (union_type is not None and origin is union_type)


def _get_resolved_type_hints(fn: Callable) -> Dict[str, Any]:
    """Resolve PEP 563 postponed annotations via get_type_hints when available."""
    try:
        return get_type_hints(fn)
    except Exception:
        return {}


def _resolve_annotation(annotation: Any) -> Any:
    """Resolve simple PEP 563 stringified annotations (e.g. 'int', 'Optional[int]', 'int | None')."""
    if not isinstance(annotation, str):
        return annotation
    text = annotation.strip()
    if text in _PRIMITIVE_ANNOTATION_NAMES:
        return _PRIMITIVE_ANNOTATION_NAMES[text]

    opt_match = re.match(r"^(?:typing\.)?Optional\[(.+)\]$", text)
    if opt_match:
        return Optional[_resolve_annotation(opt_match.group(1))]

    if "|" in text:
        parts = [p.strip() for p in text.split("|") if p.strip() not in ("None", "NoneType")]
        if len(parts) == 1:
            return Optional[_resolve_annotation(parts[0])]

    union_match = re.match(r"^(?:typing\.)?Union\[(.+)\]$", text)
    if union_match:
        parts = [p.strip() for p in union_match.group(1).split(",") if p.strip() not in ("None", "NoneType")]
        if len(parts) == 1:
            return Optional[_resolve_annotation(parts[0])]

    list_match = re.match(r"^(?:typing\.)?(?:List|list)\[(.+)\]$", text)
    if list_match:
        return List[_resolve_annotation(list_match.group(1))]

    dict_match = re.match(r"^(?:typing\.)?(?:Dict|dict)\[.+\]$", text)
    if dict_match:
        return dict

    return annotation


def _python_type_to_json_schema(annotation: Any) -> Dict[str, Any]:
    annotation = _resolve_annotation(annotation)
    if annotation is inspect.Parameter.empty or annotation is Any:
        return {"type": "string"}

    origin = get_origin(annotation)
    args = get_args(annotation)

    # Handle typing.Annotated[T, "description" / FieldInfo]
    if origin is not None and getattr(origin, "__name__", "") == "Annotated" or str(origin) == "typing.Annotated":
        if args:
            base_schema = _python_type_to_json_schema(args[0])
            for meta in args[1:]:
                if isinstance(meta, str) and "description" not in base_schema:
                    base_schema["description"] = meta
                elif hasattr(meta, "description") and getattr(meta, "description"):
                    base_schema["description"] = getattr(meta, "description")
            return base_schema

    # Handle Optional / Union (prefer BaseModel schema over raw dict when both exist)
    if _is_union_origin(origin):
        non_none = [a for a in args if a is not type(None)]
        if non_none:
            model_choice = next(
                (a for a in non_none if inspect.isclass(a) and issubclass(a, BaseModel)),
                None,
            )
            return _python_type_to_json_schema(model_choice or non_none[0])

    # Handle Literal[...]
    if getattr(origin, "__name__", "") == "Literal" or str(origin) == "typing.Literal":
        enum_vals = list(args)
        val_type = _python_type_to_json_schema(type(enum_vals[0]))["type"] if enum_vals else "string"
        return {"type": val_type, "enum": enum_vals}

    # Handle Enum subclasses
    if inspect.isclass(annotation) and issubclass(annotation, Enum):
        enum_vals = [e.value for e in annotation]
        val_type = _python_type_to_json_schema(type(enum_vals[0]))["type"] if enum_vals else "string"
        return {"type": val_type, "enum": enum_vals}

    # Handle containers
    if origin in (list, tuple, set):
        schema: Dict[str, Any] = {"type": "array"}
        if args:
            schema["items"] = _python_type_to_json_schema(args[0])
        return schema
    if origin is dict:
        return {"type": "object"}

    # Handle primitives
    if annotation is bool:
        return {"type": "boolean"}
    if annotation is int:
        return {"type": "integer"}
    if annotation is float:
        return {"type": "number"}
    if annotation in (list, tuple, set):
        return {"type": "array"}
    if annotation is dict:
        return {"type": "object"}
    if inspect.isclass(annotation) and issubclass(annotation, BaseModel):
        raw = (
            annotation.model_json_schema()
            if hasattr(annotation, "model_json_schema")
            else (annotation.schema() if hasattr(annotation, "schema") else {"type": "object"})
        )
        defs = raw.get("$defs") or raw.get("definitions") or {}
        return _resolve_json_refs(raw, defs)
    return {"type": "string"}


def _python_type_to_json_type(annotation: Any) -> str:
    return _python_type_to_json_schema(annotation).get("type", "string")


def _parse_docstring_param_descriptions(doc: Optional[str]) -> Dict[str, str]:
    """Extract parameter descriptions from Google, Sphinx, or NumPy style docstrings."""
    if not doc:
        return {}
    descriptions: Dict[str, str] = {}
    lines = doc.splitlines()
    in_args_section = False
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        # Sphinx/reST style: :param name: description
        sphinx_match = re.match(r"^:param\s+(?:\w+\s+)?(\w+):\s*(.+)$", line)
        if sphinx_match:
            descriptions[sphinx_match.group(1)] = sphinx_match.group(2).strip()
            continue

        if re.match(r"^(Args|Arguments|Parameters|Inputs):?$", line, re.IGNORECASE):
            in_args_section = True
            continue
        if re.match(r"^(Returns|Raises|Yields|Examples|Notes):?$", line, re.IGNORECASE):
            in_args_section = False
            continue

        if in_args_section:
            # Google style: name (type): description OR name: description
            arg_match = re.match(r"^(\w+)\s*(?:\([^)]*\))?\s*:\s*(.+)$", line)
            if arg_match:
                descriptions[arg_match.group(1)] = arg_match.group(2).strip()
    return descriptions


def _extract_parameters_schema(fn: Callable, args_schema: Optional[Type[BaseModel]] = None) -> Dict[str, Any]:
    if args_schema is not None and inspect.isclass(args_schema) and issubclass(args_schema, BaseModel):
        raw_schema = (
            args_schema.model_json_schema()
            if hasattr(args_schema, "model_json_schema")
            else args_schema.schema()
        )
        defs = raw_schema.get("$defs") or raw_schema.get("definitions") or {}
        resolved = _resolve_json_refs(raw_schema, defs)
        return {
            "type": "object",
            "properties": resolved.get("properties", {}),
            "required": raw_schema.get("required", []),
        }

    try:
        sig = inspect.signature(fn)
    except (ValueError, TypeError):
        return {"type": "object", "properties": {}, "required": []}

    type_hints = _get_resolved_type_hints(fn)
    doc_descriptions = _parse_docstring_param_descriptions(inspect.getdoc(fn))
    properties: Dict[str, Any] = {}
    required: List[str] = []

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue

        resolved_ann = _resolve_annotation(type_hints.get(param_name, param.annotation))
        prop_info = dict(_python_type_to_json_schema(resolved_ann))
        if param_name in doc_descriptions and "description" not in prop_info:
            prop_info["description"] = doc_descriptions[param_name]

        if param.default is not inspect.Parameter.empty:
            if param.default is not None:
                prop_info["default"] = param.default
        else:
            required.append(param_name)
        properties[param_name] = prop_info

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


def _coerce_value(value: Any, annotation: Any) -> Any:
    """Lightweight safe type coercion for LLM tool call arguments."""
    annotation = _resolve_annotation(annotation)
    if value is None or annotation is inspect.Parameter.empty or annotation is Any:
        return value

    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is not None and (getattr(origin, "__name__", "") == "Annotated" or str(origin) == "typing.Annotated"):
        if args:
            return _coerce_value(value, args[0])

    if _is_union_origin(origin):
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _coerce_value(value, non_none[0])
        return value

    try:
        if annotation is int and isinstance(value, str) and value.strip().lstrip("-").isdigit():
            return int(value.strip())
        if annotation is float and isinstance(value, (str, int)) and not isinstance(value, bool):
            return float(value)
        if annotation is bool and isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in ("true", "1", "yes"):
                return True
            if lowered in ("false", "0", "no"):
                return False
        if inspect.isclass(annotation) and issubclass(annotation, BaseModel) and isinstance(value, dict):
            return annotation.model_validate(value) if hasattr(annotation, "model_validate") else annotation(**value)
    except Exception:
        return value
    return value


def _apply_param_aliases(
    raw_kwargs: Dict[str, Any],
    param_aliases: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Map legacy/alternative parameter names to canonical names when canonical key is absent."""
    kwargs = dict(raw_kwargs or {})
    if not param_aliases:
        return kwargs
    for alias, canonical in param_aliases.items():
        if alias in kwargs and kwargs[alias] is not None:
            if canonical not in kwargs or kwargs[canonical] is None:
                kwargs[canonical] = kwargs.pop(alias)
            else:
                kwargs.pop(alias, None)
    return kwargs


def _prepare_call_kwargs(
    fn: Callable,
    raw_kwargs: Dict[str, Any],
    args_schema: Optional[Type[BaseModel]] = None,
    param_aliases: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Validate and bind kwargs to a callable's signature, resolving aliases and filtering unexpected keys."""
    kwargs = _apply_param_aliases(raw_kwargs, param_aliases=param_aliases)

    if args_schema is not None and inspect.isclass(args_schema) and issubclass(args_schema, BaseModel):
        validated = (
            args_schema.model_validate(kwargs)
            if hasattr(args_schema, "model_validate")
            else args_schema(**kwargs)
        )
        kwargs = (
            validated.model_dump()
            if hasattr(validated, "model_dump")
            else validated.dict()
        )
        extra = getattr(validated, "model_extra", None) or getattr(validated, "__pydantic_extra__", None)
        if isinstance(extra, dict):
            for ek, ev in extra.items():
                if ek not in kwargs:
                    kwargs[ek] = ev

    try:
        sig = inspect.signature(fn)
    except (ValueError, TypeError):
        return kwargs

    type_hints = _get_resolved_type_hints(fn)
    accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    bound: Dict[str, Any] = {}
    for k, v in kwargs.items():
        if k in sig.parameters:
            param = sig.parameters[k]
            ann = _resolve_annotation(type_hints.get(k, param.annotation))
            bound[k] = _coerce_value(v, ann)
        elif accepts_var_kw:
            bound[k] = v

    return bound


def _normalize_tool_output(result: Any) -> ToolResult:
    """Normalize any return value from a custom tool into a standard ToolResult."""
    if isinstance(result, ToolResult):
        return result
    if isinstance(result, dict) and "success" in result and ("output" in result or "error" in result):
        return ToolResult(
            success=bool(result.get("success", True)),
            output=result.get("output"),
            error=result.get("error"),
        )
    if isinstance(result, BaseModel):
        dumped = result.model_dump() if hasattr(result, "model_dump") else result.dict()
        return ToolResult(success=True, output=dumped)
    return ToolResult(success=True, output=result)


class BaseTool:
    name: str = ""
    description: str = ""
    args_schema: Optional[Type[BaseModel]] = None
    param_aliases: Dict[str, str] = {}
    read_only: bool = False
    return_direct: bool = False
    timeout: Optional[float] = None
    runtime: Optional[Any] = None

    def __init__(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
        args_schema: Optional[Type[BaseModel]] = None,
        read_only: Optional[bool] = None,
        return_direct: Optional[bool] = None,
        runtime: Optional[Any] = None,
        timeout: Optional[float] = None,
    ):
        if name is not None:
            self.name = name
        elif not getattr(self, "name", None):
            self.name = self.__class__.__name__.lower()

        if description is not None:
            self.description = description
        elif not getattr(self, "description", None):
            self.description = inspect.getdoc(self) or f"Executes the {self.name} tool."

        if args_schema is not None:
            self.args_schema = args_schema
        if read_only is not None:
            self.read_only = read_only
        if return_direct is not None:
            self.return_direct = return_direct
        if runtime is not None:
            self.runtime = runtime
        if timeout is not None:
            self.timeout = timeout

    def set_runtime(self, runtime: Any) -> None:
        self.runtime = runtime
        self._default_runtime_created = False

    async def execute(self, **kwargs) -> Union[ToolResult, Any]:
        raise NotImplementedError("Tool must implement execute method")

    async def invoke(self, **kwargs) -> ToolResult:
        """
        Safe execution entry point that supports both async and sync `execute` implementations,
        validates/coerces parameters, enforces optional timeout, and normalizes any return value into `ToolResult`.
        """
        target_fn = getattr(self, "func", self.execute)
        aliases = getattr(self, "param_aliases", None)
        try:
            call_kwargs = _prepare_call_kwargs(
                target_fn,
                kwargs,
                args_schema=getattr(self, "args_schema", None),
                param_aliases=aliases,
            )

            async def _run_inner() -> ToolResult:
                exec_method = self.execute
                if inspect.iscoroutinefunction(exec_method):
                    raw_res = await exec_method(**call_kwargs)
                else:
                    maybe_coro = await asyncio.to_thread(exec_method, **call_kwargs)
                    raw_res = await maybe_coro if inspect.isawaitable(maybe_coro) else maybe_coro
                return _normalize_tool_output(raw_res)

            timeout_val = getattr(self, "timeout", None)
            if timeout_val is not None and timeout_val > 0:
                return await asyncio.wait_for(_run_inner(), timeout=timeout_val)
            return await _run_inner()
        except asyncio.TimeoutError:
            return ToolResult(
                success=False,
                output=None,
                error=f"Tool '{self.name}' timed out after {getattr(self, 'timeout', '?')}s.",
            )
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))

    async def run(self, **kwargs) -> Any:
        result = await self.invoke(**kwargs)
        return result.output if result.success else result.error

    def get_parameters_schema(self) -> Dict[str, Any]:
        target_fn = getattr(self, "func", self.execute)
        return _extract_parameters_schema(target_fn, args_schema=getattr(self, "args_schema", None))

    def schema(self) -> Dict[str, Any]:
        return self.to_dict()

    def to_dict(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "parameters": self.get_parameters_schema(),
        }
        if getattr(self, "read_only", False):
            info["read_only"] = True
        if getattr(self, "return_direct", False):
            info["return_direct"] = True
        return info

    def to_openai_tool(self) -> Dict[str, Any]:
        """Convert this tool into OpenAI Chat Completions `tools` function schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description or f"Executes {self.name}",
                "parameters": self.get_parameters_schema(),
            },
        }


class FunctionTool(BaseTool):
    """
    A BaseTool backed by any synchronous or asynchronous Python function.
    """

    def __init__(
        self,
        func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None,
        args_schema: Optional[Type[BaseModel]] = None,
        read_only: bool = False,
        return_direct: bool = False,
        runtime: Optional[Any] = None,
    ):
        resolved_name = (name or getattr(func, "__name__", "custom_tool")).strip()
        resolved_desc = (
            description
            or inspect.getdoc(func)
            or f"Custom tool '{resolved_name}'."
        ).strip()
        super().__init__(
            name=resolved_name,
            description=resolved_desc,
            args_schema=args_schema,
            read_only=read_only,
            return_direct=return_direct,
            runtime=runtime,
        )
        self.func = func

    @classmethod
    def from_function(
        cls,
        func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None,
        args_schema: Optional[Type[BaseModel]] = None,
        read_only: bool = False,
        return_direct: bool = False,
        runtime: Optional[Any] = None,
    ) -> "FunctionTool":
        return cls(
            func=func,
            name=name,
            description=description,
            args_schema=args_schema,
            read_only=read_only,
            return_direct=return_direct,
            runtime=runtime,
        )

    async def execute(self, **kwargs) -> ToolResult:
        try:
            call_kwargs = _prepare_call_kwargs(
                self.func,
                kwargs,
                args_schema=self.args_schema,
                param_aliases=getattr(self, "param_aliases", None),
            )
            if inspect.iscoroutinefunction(self.func):
                result = await self.func(**call_kwargs)
            else:
                maybe_coro = await asyncio.to_thread(self.func, **call_kwargs)
                result = await maybe_coro if inspect.isawaitable(maybe_coro) else maybe_coro
            return _normalize_tool_output(result)
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))

    def __call__(self, *args, **kwargs):
        """Allow the decorated function to still be called directly in Python."""
        return self.func(*args, **kwargs)


CustomTool = FunctionTool


def tool(
    name_or_func: Optional[Union[str, Callable]] = None,
    description: Optional[str] = None,
    *,
    name: Optional[str] = None,
    args_schema: Optional[Type[BaseModel]] = None,
    read_only: bool = False,
    return_direct: bool = False,
):
    """
    Flexible decorator to convert any sync or async Python function into a BaseTool.

    Supported usages:
      1. @tool("say_hello", "Greets a user by name")
      2. @tool("say_hello")  # description inferred from docstring
      3. @tool               # name & description inferred from function name and docstring
      4. @tool(name="say_hello", description="...", args_schema=..., read_only=True)
    """
    # Case 3: @tool without parentheses
    if callable(name_or_func):
        fn = name_or_func
        return FunctionTool(
            func=fn,
            name=name,
            description=description,
            args_schema=args_schema,
            read_only=read_only,
            return_direct=return_direct,
        )

    resolved_name = name or (name_or_func if isinstance(name_or_func, str) else None)

    def decorator(func: Callable) -> FunctionTool:
        return FunctionTool(
            func=func,
            name=resolved_name,
            description=description,
            args_schema=args_schema,
            read_only=read_only,
            return_direct=return_direct,
        )

    return decorator


def create_tool(
    func: Callable,
    name: Optional[str] = None,
    description: Optional[str] = None,
    args_schema: Optional[Type[BaseModel]] = None,
    read_only: bool = False,
    return_direct: bool = False,
) -> FunctionTool:
    """Create a BaseTool programmatically from any sync or async callable."""
    return FunctionTool(
        func=func,
        name=name,
        description=description,
        args_schema=args_schema,
        read_only=read_only,
        return_direct=return_direct,
    )


def ensure_tool(
    item: Any,
    name: Optional[str] = None,
    description: Optional[str] = None,
    **kwargs: Any,
) -> BaseTool:
    """
    Normalize any user-provided tool specification into a `BaseTool` instance.
    Accepts:
      - A `BaseTool` instance
      - A `BaseTool` subclass (auto-instantiated)
      - A plain Python function / callable (sync or async)
      - A dict with `{"func": callable, "name": "...", "description": "..."}`
    """
    if isinstance(item, BaseTool):
        if name:
            item.name = name
        if description:
            item.description = description
        return item

    if inspect.isclass(item) and issubclass(item, BaseTool):
        inst = item()
        if name:
            inst.name = name
        if description:
            inst.description = description
        return inst

    if isinstance(item, dict) and callable(item.get("func")):
        return FunctionTool(
            func=item["func"],
            name=name or item.get("name"),
            description=description or item.get("description"),
            args_schema=kwargs.get("args_schema") or item.get("args_schema"),
            read_only=bool(kwargs.get("read_only", item.get("read_only", False))),
            return_direct=bool(kwargs.get("return_direct", item.get("return_direct", False))),
        )

    if callable(item):
        return FunctionTool(
            func=item,
            name=name,
            description=description,
            args_schema=kwargs.get("args_schema"),
            read_only=bool(kwargs.get("read_only", False)),
            return_direct=bool(kwargs.get("return_direct", False)),
        )

    raise TypeError(
        f"Cannot convert {type(item).__name__} to BaseTool. "
        "Provide a BaseTool instance/subclass, a @tool function, a Python callable, or a tool dict."
    )

