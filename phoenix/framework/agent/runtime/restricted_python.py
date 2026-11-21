import ast
import builtins as _host_builtins
import io
import os
import shutil
import asyncio
from typing import Optional
from .base import BaseRuntime, ExecutionResult


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
        "globals", "locals", "vars", "breakpoint", "input",
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
        "True", "False", "None",
    )

    def __init__(
        self,
        workspace_root: Optional[str] = None,
        timeout: float = 30.0,
        max_output_chars: int = 24000,
    ):
        self.workspace_root = os.path.realpath(workspace_root) if workspace_root else None
        self.timeout = timeout
        self.max_output_chars = max(500, int(max_output_chars))

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

    def _build_safe_builtins(self, output_buffer: io.StringIO) -> dict:
        safe_b = {
            name: getattr(_host_builtins, name)
            for name in self.SAFE_BUILTIN_NAMES
            if hasattr(_host_builtins, name)
        }

        def _safe_print(*args, sep=" ", end="\n", **_kwargs):
            output_buffer.write(sep.join(str(a) for a in args) + end)

        def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
            base_module = name.split(".")[0]
            if base_module in self.DANGEROUS_MODULES or base_module not in self.ALLOWED_MODULES:
                raise ImportError(f"Importing '{base_module}' is blocked in RestrictedPythonRuntime.")
            return _host_builtins.__import__(name, globals, locals, fromlist, level)

        safe_b["print"] = _safe_print
        safe_b["__import__"] = _safe_import
        safe_b["__build_class__"] = _host_builtins.__build_class__
        return safe_b

    async def execute_code(self, code: str, language: str = "python") -> ExecutionResult:
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

        output_buffer = io.StringIO()
        safe_builtins = self._build_safe_builtins(output_buffer)
        exec_globals = {"__builtins__": safe_builtins, "__name__": "__main__"}

        def _run():
            # Share module namespace so top-level imports & helper functions stay visible inside functions
            exec(code, exec_globals, exec_globals)

        try:
            await asyncio.wait_for(asyncio.to_thread(_run), timeout=self.timeout)
        except asyncio.TimeoutError:
            return ExecutionResult(
                success=False,
                output=output_buffer.getvalue(),
                error=f"Execution timed out after {self.timeout}s",
                exit_code=124,
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                output=output_buffer.getvalue(),
                error=f"Execution error: {str(e)}",
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

    async def execute_command(self, command: str, cwd: Optional[str] = None) -> ExecutionResult:
        """
        Executes a shell command with dangerous-pattern filtering and timeout enforcement.
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

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=effective_cwd,
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=self.timeout)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.communicate()
                return ExecutionResult(
                    success=False,
                    output="",
                    error=f"Command timed out after {self.timeout}s",
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
