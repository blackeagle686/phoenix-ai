import ast
import builtins as _host_builtins
import io
import os
import shutil
import signal
import asyncio
import threading
import time as _host_time
import traceback
import types
from typing import Any, List, Optional
from .base import BaseRuntime, ExecutionResult


class _SandboxAbortBase(BaseException):
    """Internal cooperative abort signal that bypasses user `except Exception:` handlers."""


class SandboxTimeoutError(_SandboxAbortBase):
    pass


class InfiniteLoopError(_SandboxAbortBase):
    pass


class InteractiveLoopError(_SandboxAbortBase):
    pass


class _LoopGuardTransformer(ast.NodeTransformer):
    """Injects cooperative cancellation, deadline, and iteration checks into every `while` loop."""

    def visit_While(self, node: ast.While) -> ast.AST:
        self.generic_visit(node)
        tick_call = ast.Expr(
            value=ast.Call(
                func=ast.Name(id="__phoenix_loop_tick__", ctx=ast.Load()),
                args=[ast.Constant(value=int(getattr(node, "lineno", 0) or 0))],
                keywords=[],
            )
        )
        ast.copy_location(tick_call, node)
        ast.copy_location(tick_call.value, node)
        ast.copy_location(tick_call.value.func, node)
        ast.copy_location(tick_call.value.args[0], node)
        node.body.insert(0, tick_call)
        return node


def _choose_simulated_input(prompt_text: str, call_idx: int) -> str:
    """Return a non-blocking deterministic input line suited to CLI calculators, menus, or general prompts."""
    if call_idx == 0:
        p_low = (prompt_text or "").lower()
        if any(k in p_low for k in ("choice", "option", "menu", "select")):
            return "1"
        if any(k in p_low for k in ("expr", "formula", "equation", "calc", "math")):
            return "2 + 3"
        if any(k in p_low for k in ("num", "int", "float", "age", "count", "value", "first", "second")):
            return "2"
        return "2"
    fallback_seq = ("quit", "exit", "q", "0")
    seq_idx = call_idx - 1
    return fallback_seq[seq_idx] if seq_idx < len(fallback_seq) else "quit"


class RestrictedPythonRuntime(BaseRuntime):
    """
    Zero-dependency runtime that uses AST filtering, restricted builtins,
    concurrency-safe output capture, and optional workspace path jailing.
    """

    DANGEROUS_MODULES = {
        "os", "sys", "subprocess", "shutil", "pty", "socket", "builtins",
        "macpath", "ntpath", "posixpath", "cmd", "shlex", "importlib",
        "ctypes", "pickle", "marshal", "pathlib", "tempfile", "glob",
    }

    ALLOWED_MODULES = {
        "math", "json", "re", "datetime", "collections", "itertools",
        "functools", "statistics", "random", "decimal", "fractions",
        "hashlib", "base64", "uuid", "string", "typing", "dataclasses",
        "enum", "copy", "bisect", "heapq", "time", "operator", "ast",
    }

    BLOCKED_FUNCTIONS = {
        "eval", "exec", "open", "compile", "__import__",
        "globals", "locals", "vars", "breakpoint",
    }

    BLOCKED_DUNDERS = {
        "__subclasses__", "__bases__", "__mro__", "__globals__",
        "__builtins__", "__import__", "__code__", "__closure__",
    }

    SAFE_BUILTIN_NAMES = (
        "abs", "all", "any", "ascii", "bin", "bool", "bytearray", "bytes",
        "callable", "chr", "complex", "dict", "divmod", "enumerate", "filter",
        "float", "format", "frozenset", "hash", "hex", "id", "int",
        "isinstance", "issubclass", "iter", "len", "list", "map", "max",
        "min", "next", "object", "oct", "ord", "pow", "property", "range",
        "repr", "reversed", "round", "set", "slice", "sorted", "staticmethod",
        "str", "sum", "super", "tuple", "type", "zip",
        "Exception", "ValueError", "TypeError", "KeyError", "IndexError",
        "ZeroDivisionError", "RuntimeError", "StopIteration", "AssertionError",
        "NameError", "AttributeError", "ImportError", "ArithmeticError", "OverflowError",
        "EOFError", "KeyboardInterrupt",
        "True", "False", "None",
    )

    def __init__(
        self,
        workspace_root: Optional[str] = None,
        timeout: float = 30.0,
        max_output_chars: int = 24000,
        max_loop_iterations: int = 25000,
        interactive_input_limit: int = 5,
    ):
        self.workspace_root = os.path.realpath(workspace_root) if workspace_root else None
        self.timeout = timeout
        self.max_output_chars = max(500, int(max_output_chars))
        self.max_loop_iterations = max(100, int(max_loop_iterations))
        self.interactive_input_limit = max(1, int(interactive_input_limit))

    def _cap_text(self, text: str) -> str:
        if not text or len(text) <= self.max_output_chars:
            return text
        head = int(self.max_output_chars * 0.65)
        tail = self.max_output_chars - head - 40
        return f"{text[:head]}\n... [output truncated] ...\n{text[-max(0, tail):]}"

    def _is_safe_code(self, code: str) -> tuple[bool, Optional[str]]:
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, f"Syntax Error: {str(e)}"

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    base_module = alias.name.split(".")[0]
                    if base_module in self.DANGEROUS_MODULES:
                        return False, f"Importing '{base_module}' is blocked for security reasons."
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    base_module = node.module.split(".")[0]
                    if base_module in self.DANGEROUS_MODULES:
                        return False, f"Importing from '{base_module}' is blocked for security reasons."
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in self.BLOCKED_FUNCTIONS:
                        return False, f"Function '{node.func.id}' is blocked for security reasons."
            elif isinstance(node, ast.Attribute):
                if node.attr in self.BLOCKED_DUNDERS:
                    return False, f"Attribute '{node.attr}' is blocked for security reasons."

        return True, None

    def _build_safe_builtins(
        self,
        output_buffer: io.StringIO,
        *,
        cancel_event: Optional[threading.Event] = None,
        deadline: Optional[float] = None,
        max_loop_iterations: Optional[int] = None,
        stdin_lines: Optional[List[str]] = None,
    ) -> dict:
        safe_b = {
            name: getattr(_host_builtins, name)
            for name in self.SAFE_BUILTIN_NAMES
            if hasattr(_host_builtins, name)
        }
        ev = cancel_event or threading.Event()
        max_iters = self.max_loop_iterations if max_loop_iterations is None else max_loop_iterations
        input_limit = self.interactive_input_limit
        explicit_iter = iter(stdin_lines) if stdin_lines is not None else None
        state = {"loop_ticks": 0, "input_calls": 0, "eof_calls": 0}

        def _check_deadline(lineno: int = 0) -> None:
            loc_suffix = f" at line {lineno}" if lineno else ""
            if ev.is_set() or (deadline is not None and _host_time.monotonic() > deadline):
                raise SandboxTimeoutError(f"Execution timed out{loc_suffix}")

        def _loop_tick(lineno: int = 0) -> None:
            _check_deadline(lineno)
            state["loop_ticks"] += 1
            if max_iters > 0 and state["loop_ticks"] > max_iters:
                raise InfiniteLoopError(
                    f"Infinite/runaway while loop detected at line {lineno} (exceeded {max_iters} iterations)"
                )

        def _safe_input(prompt: Any = "") -> str:
            _check_deadline()
            if prompt:
                output_buffer.write(str(prompt))
            if explicit_iter is not None:
                try:
                    val = str(next(explicit_iter))
                    output_buffer.write(val + "\n")
                    return val
                except StopIteration:
                    state["eof_calls"] += 1
                    if state["eof_calls"] > 2:
                        raise InteractiveLoopError(
                            "Interactive loop kept calling input() repeatedly after scripted stdin reached EOF."
                        )
                    raise EOFError("End of scripted input (EOF)")

            call_idx = state["input_calls"]
            if call_idx >= input_limit:
                state["eof_calls"] += 1
                if state["eof_calls"] > 2:
                    raise InteractiveLoopError(
                        "Interactive input loop did not terminate after simulated inputs ('quit'/'exit') and EOF."
                    )
                raise EOFError("No interactive stdin in sandbox (EOF)")

            state["input_calls"] += 1
            sim_val = _choose_simulated_input(str(prompt or ""), call_idx)
            output_buffer.write(sim_val + "\n")
            return sim_val

        def _cooperative_sleep(seconds: float) -> None:
            secs = max(0.0, float(seconds))
            _check_deadline()
            if deadline is None:
                if ev.wait(timeout=secs):
                    raise SandboxTimeoutError("Execution cancelled during sleep")
                return
            remaining = deadline - _host_time.monotonic()
            if remaining <= 0:
                raise SandboxTimeoutError("Execution timed out during sleep")
            wait_for = min(secs, remaining)
            if ev.wait(timeout=wait_for):
                raise SandboxTimeoutError("Execution cancelled during sleep")
            if secs > remaining:
                raise SandboxTimeoutError("Execution timed out during sleep")

        def _wrap_time_module(mod: types.ModuleType) -> types.ModuleType:
            proxy = types.ModuleType("time")
            for attr in dir(mod):
                if attr != "sleep":
                    try:
                        setattr(proxy, attr, getattr(mod, attr))
                    except Exception:
                        pass
            proxy.sleep = _cooperative_sleep
            return proxy

        def _safe_print(*args, sep=" ", end="\n", **_kwargs):
            _check_deadline()
            output_buffer.write(sep.join(str(a) for a in args) + end)

        def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
            base_module = name.split(".")[0]
            if base_module in self.DANGEROUS_MODULES or base_module not in self.ALLOWED_MODULES:
                raise ImportError(f"Importing '{base_module}' is blocked in RestrictedPythonRuntime.")
            mod = _host_builtins.__import__(name, globals, locals, fromlist, level)
            if name == "time" or base_module == "time":
                return _wrap_time_module(mod)
            return mod

        def _safe_hasattr(obj, name):
            if not isinstance(name, str) or name.startswith("_") or name in self.BLOCKED_DUNDERS:
                return False
            return _host_builtins.hasattr(obj, name)

        def _safe_getattr(obj, name, *default):
            if not isinstance(name, str) or name.startswith("_") or name in self.BLOCKED_DUNDERS:
                raise AttributeError(f"Access to private/restricted attribute '{name}' is blocked.")
            return _host_builtins.getattr(obj, name, *default)

        safe_b["print"] = _safe_print
        safe_b["input"] = _safe_input
        safe_b["__phoenix_loop_tick__"] = _loop_tick
        safe_b["hasattr"] = _safe_hasattr
        safe_b["getattr"] = _safe_getattr
        safe_b["__import__"] = _safe_import
        safe_b["__build_class__"] = _host_builtins.__build_class__
        return safe_b

    async def execute_code(
        self,
        code: str,
        language: str = "python",
        *,
        timeout: Optional[float] = None,
        max_loop_iterations: Optional[int] = None,
        stdin_lines: Optional[List[str]] = None,
    ) -> ExecutionResult:
        if language.lower() != "python":
            return ExecutionResult(success=False, output="", error=f"Unsupported language: {language}", exit_code=1)

        is_safe, error_msg = self._is_safe_code(code)
        if not is_safe:
            return ExecutionResult(
                success=False,
                output="",
                error=f"Security Violation: {error_msg}",
                exit_code=1,
            )

        eff_timeout = float(timeout if timeout is not None else self.timeout)
        eff_max_iters = self.max_loop_iterations if max_loop_iterations is None else int(max_loop_iterations)
        cancel_event = threading.Event()
        deadline = _host_time.monotonic() + eff_timeout if eff_timeout > 0 else None

        output_buffer = io.StringIO()
        safe_builtins = self._build_safe_builtins(
            output_buffer,
            cancel_event=cancel_event,
            deadline=deadline,
            max_loop_iterations=eff_max_iters,
            stdin_lines=stdin_lines,
        )
        exec_globals = {"__builtins__": safe_builtins, "__name__": "__main__"}
        abort_holder: dict = {"err": None}

        def _run():
            try:
                tree = ast.parse(code)
                guarded_tree = _LoopGuardTransformer().visit(tree)
                ast.fix_missing_locations(guarded_tree)
                compiled = compile(guarded_tree, "<sandbox>", "exec")
                exec(compiled, exec_globals, exec_globals)
            except _SandboxAbortBase as abort_exc:
                abort_holder["err"] = abort_exc

        try:
            await asyncio.wait_for(asyncio.to_thread(_run), timeout=eff_timeout)
        except asyncio.TimeoutError:
            cancel_event.set()
            return ExecutionResult(
                success=False,
                output=output_buffer.getvalue(),
                error=f"Execution timed out after {eff_timeout}s",
                exit_code=124,
            )
        except Exception as e:
            cancel_event.set()
            tb_frames = [
                fr for fr in traceback.extract_tb(e.__traceback__)
                if fr.filename == "<sandbox>"
            ]
            loc = f" at line {tb_frames[-1].lineno}" if tb_frames else ""
            chain = ""
            if len(tb_frames) > 1:
                chain = " [trace: " + " -> ".join(f"{fr.name}@L{fr.lineno}" for fr in tb_frames[-4:]) + "]"
            return ExecutionResult(
                success=False,
                output=output_buffer.getvalue(),
                error=f"Execution error ({type(e).__name__}{loc}{chain}): {str(e)}",
                exit_code=1,
            )
        finally:
            cancel_event.set()

        if abort_holder["err"] is not None:
            ab = abort_holder["err"]
            if isinstance(ab, SandboxTimeoutError):
                return ExecutionResult(
                    success=False,
                    output=output_buffer.getvalue(),
                    error=f"Execution timed out after {eff_timeout}s",
                    exit_code=124,
                )
            tb_frames = [
                fr for fr in traceback.extract_tb(ab.__traceback__)
                if fr.filename == "<sandbox>"
            ]
            loc = f" at line {tb_frames[-1].lineno}" if tb_frames else ""
            chain = ""
            if len(tb_frames) > 1:
                chain = " [trace: " + " -> ".join(f"{fr.name}@L{fr.lineno}" for fr in tb_frames[-4:]) + "]"
            return ExecutionResult(
                success=False,
                output=output_buffer.getvalue(),
                error=f"Execution error ({type(ab).__name__}{loc}{chain}): {str(ab)}",
                exit_code=1,
            )

        output = self._cap_text(output_buffer.getvalue())
        if not output and "result" in exec_globals:
            output = self._cap_text(str(exec_globals["result"]))

        return ExecutionResult(
            success=True,
            output=output.strip() if output else "Executed successfully (no output)",
            error=None,
            exit_code=0,
        )

    async def execute_command(
        self,
        command: str,
        cwd: Optional[str] = None,
        *,
        stdin_data: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> ExecutionResult:
        """
        Executes a shell command with dangerous-pattern filtering, non-blocking stdin, and timeout enforcement.
        """
        dangerous_patterns = [
            "rm -rf /",
            "mkfs",
            ":(){ :|:& };:",
            "dd if=/dev/zero",
            "> /dev/sda",
            "chmod -R 777 /",
        ]
        for pattern in dangerous_patterns:
            if pattern in command:
                return ExecutionResult(
                    success=False,
                    output="",
                    error="Security Violation: Command contains dangerous pattern.",
                    exit_code=1,
                )

        effective_cwd = cwd or self.workspace_root
        if effective_cwd and self.workspace_root:
            real_cwd = os.path.realpath(effective_cwd)
            if os.path.commonpath([self.workspace_root, real_cwd]) != self.workspace_root:
                return ExecutionResult(
                    success=False,
                    output="",
                    error=f"Security Violation: Working directory '{cwd}' is outside workspace root.",
                    exit_code=1,
                )

        eff_timeout = float(timeout if timeout is not None else self.timeout)
        sub_kwargs: dict = {
            "stdin": asyncio.subprocess.PIPE if stdin_data is not None else asyncio.subprocess.DEVNULL,
            "stdout": asyncio.subprocess.PIPE,
            "stderr": asyncio.subprocess.PIPE,
            "cwd": effective_cwd,
        }
        if os.name == "posix":
            sub_kwargs["start_new_session"] = True

        try:
            proc = await asyncio.create_subprocess_shell(command, **sub_kwargs)
            in_bytes = stdin_data.encode("utf-8") if stdin_data is not None else None
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(input=in_bytes), timeout=eff_timeout)
            except asyncio.TimeoutError:
                try:
                    if os.name == "posix" and proc.pid:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    else:
                        proc.kill()
                except ProcessLookupError:
                    pass
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                await proc.communicate()
                return ExecutionResult(
                    success=False,
                    output="",
                    error=f"Command timed out after {eff_timeout}s",
                    exit_code=124,
                )

            stdout_str = self._cap_text(stdout.decode("utf-8", errors="replace"))
            stderr_str = self._cap_text(stderr.decode("utf-8", errors="replace"))
            exit_code = proc.returncode or 0
            success = exit_code == 0

            return ExecutionResult(
                success=success,
                output=stdout_str,
                error=stderr_str if not success else None,
                exit_code=exit_code,
            )
        except Exception as e:
            return ExecutionResult(success=False, output="", error=str(e), exit_code=1)

    def _resolve_and_check_path(self, file_path: str) -> tuple[bool, str, Optional[str]]:
        base = self.workspace_root or os.getcwd()
        candidate = file_path if os.path.isabs(file_path) else os.path.join(base, file_path)
        abs_path = os.path.abspath(candidate)
        if self.workspace_root:
            try:
                real_path = os.path.realpath(abs_path)
                if os.path.commonpath([self.workspace_root, real_path]) != self.workspace_root:
                    return (
                        False,
                        abs_path,
                        f"Security Violation: Path '{file_path}' resolves outside workspace root '{self.workspace_root}'.",
                    )
            except ValueError:
                return False, abs_path, f"Security Violation: Invalid path '{file_path}'."
        return True, abs_path, None

    async def execute_io(self, operation: str, file_path: str, content: Optional[str] = None) -> ExecutionResult:
        """
        Executes File I/O operations natively and safely without blocking the event loop.
        """
        allowed, abs_path, err = self._resolve_and_check_path(file_path)
        if not allowed:
            return ExecutionResult(success=False, output="", error=err, exit_code=1)

        def _sync_io() -> ExecutionResult:
            try:
                if operation == "create_dir":
                    os.makedirs(abs_path, exist_ok=True)
                    return ExecutionResult(success=True, output=f"Successfully created directory {file_path}")

                elif operation in ("create", "edit", "write", "append"):
                    text_content = content if content is not None else ""
                    parent = os.path.dirname(abs_path)
                    if parent:
                        os.makedirs(parent, exist_ok=True)

                    mode = "a" if operation == "append" else "w"
                    with open(abs_path, mode, encoding="utf-8", newline="") as f:
                        f.write(text_content)
                    return ExecutionResult(success=True, output=f"Successfully executed {operation} on {file_path}")

                elif operation == "read":
                    if not os.path.exists(abs_path):
                        return ExecutionResult(success=False, output="", error=f"File not found: {file_path}", exit_code=1)
                    with open(abs_path, "r", encoding="utf-8", newline="") as f:
                        data = f.read()
                    return ExecutionResult(success=True, output=data)

                elif operation == "delete":
                    if not os.path.exists(abs_path):
                        return ExecutionResult(success=False, output="", error=f"File not found: {file_path}", exit_code=1)
                    if os.path.isdir(abs_path):
                        shutil.rmtree(abs_path)
                    else:
                        os.remove(abs_path)
                    return ExecutionResult(success=True, output=f"Successfully deleted {file_path}")

                else:
                    return ExecutionResult(success=False, output="", error=f"Unknown IO operation: {operation}", exit_code=1)
            except Exception as e:
                return ExecutionResult(success=False, output="", error=str(e), exit_code=1)

        return await asyncio.to_thread(_sync_io)
