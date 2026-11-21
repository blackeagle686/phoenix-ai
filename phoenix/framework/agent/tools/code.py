"""Safe Python REPL, AST analyzer, sandboxed command execution, and syntax compiler tools."""

from __future__ import annotations

import ast
import asyncio
import py_compile
from typing import Any, Optional

from phoenix.framework.agent.cognition.schema import (
    CodeCompileResult,
    CodeExecutionResult,
    CommandExecutionResult,
    PythonAnalysisResult,
    PythonClassInfo,
    PythonMethodInfo,
)
from phoenix.framework.agent.runtime.restricted_python import RestrictedPythonRuntime
from phoenix.framework.agent.tools._fs_utils import resolve_existing_path
from phoenix.framework.agent.tools.base import BaseTool, ToolResult


def _dump_model(model: Any) -> dict:
    return model.model_dump() if hasattr(model, "model_dump") else model.dict()


class CodeExecutionTool(BaseTool):
    name = "python_repl"
    description = "Executes sandboxed Python code (math, json, re, collections, etc.) and captures output. Input: 'code' (str)."

    def __init__(self, runtime: Optional[Any] = None):
        super().__init__()
        self._default_runtime_created = runtime is None
        self.runtime = runtime or RestrictedPythonRuntime()

    async def execute(self, code: str, **kwargs) -> ToolResult:
        try:
            if not code or not str(code).strip():
                return ToolResult(success=False, output=None, error="Empty Python code passed to python_repl.")
            exe_result = await self.runtime.execute_code(code, language="python")
            result = CodeExecutionResult(
                success=exe_result.success,
                output=exe_result.output,
                error=exe_result.error,
            )
            return ToolResult(success=exe_result.success, output=_dump_model(result), error=exe_result.error)
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))


class PythonAnalyzerTool(BaseTool):
    """Fast AST-based analyzer for Python files inside the workspace."""

    name = "python_analyzer"
    read_only = True
    param_aliases = {"path": "file_path"}
    description = "Analyzes a Python file inside the workspace and returns classes, functions, line numbers, and imports. Input: 'file_path' (str)."

    def __init__(self, runtime: Optional[Any] = None):
        super().__init__(read_only=True)
        self._default_runtime_created = runtime is None
        self.runtime = runtime or RestrictedPythonRuntime()

    async def execute(self, file_path: str, **kwargs) -> ToolResult:
        try:
            ok, resolved_path, err, meta = await asyncio.to_thread(
                resolve_existing_path, file_path, self.runtime, False
            )
            if not ok:
                return ToolResult(success=False, output=meta or None, error=err)

            read_res = await self.runtime.execute_io("read", resolved_path)
            if not read_res.success:
                return ToolResult(success=False, output=None, error=read_res.error)

            source = read_res.output or ""

            def _parse_ast() -> PythonAnalysisResult:
                tree = ast.parse(source)
                classes_map = {}
                functions_list = []
                imports_list = []

                for node in tree.body:
                    if isinstance(node, ast.ClassDef):
                        methods = [
                            PythonMethodInfo(
                                name=n.name,
                                line_start=n.lineno,
                                line_end=getattr(n, "end_lineno", n.lineno),
                            )
                            for n in node.body
                            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                        ]
                        classes_map[node.name] = PythonClassInfo(
                            line_start=node.lineno,
                            line_end=getattr(node, "end_lineno", node.lineno),
                            methods=methods,
                        )
                    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        functions_list.append(
                            PythonMethodInfo(
                                name=node.name,
                                line_start=node.lineno,
                                line_end=getattr(node, "end_lineno", node.lineno),
                            )
                        )
                    elif isinstance(node, (ast.Import, ast.ImportFrom)):
                        imports_list.append(ast.unparse(node))

                return PythonAnalysisResult(
                    file_path=resolved_path,
                    classes=classes_map,
                    functions=functions_list,
                    imports=imports_list,
                )

            result = await asyncio.to_thread(_parse_ast)
            out = _dump_model(result)
            out.update(meta)
            return ToolResult(success=True, output=out)
        except SyntaxError as e:
            return ToolResult(success=False, output=None, error=f"Syntax error in Python file: {e}")
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))


class CommandExecutionTool(BaseTool):
    name = "execute_command"
    description = "Executes a shell/terminal command inside the workspace and returns stdout, stderr, and exit code. Inputs: 'command' (str), optional 'cwd' (str)."

    def __init__(self, runtime: Optional[Any] = None):
        super().__init__()
        self._default_runtime_created = runtime is None
        self.runtime = runtime or RestrictedPythonRuntime()

    async def execute(self, command: str, cwd: Optional[str] = None, **kwargs) -> ToolResult:
        try:
            if not command or not str(command).strip():
                return ToolResult(success=False, output=None, error="Empty command passed to execute_command.")
            exe_result = await self.runtime.execute_command(command, cwd=cwd)
            result = CommandExecutionResult(
                command=command,
                success=exe_result.success,
                stdout=exe_result.output,
                stderr=exe_result.error or "",
                exit_code=exe_result.exit_code,
            )
            return ToolResult(success=exe_result.success, output=_dump_model(result), error=exe_result.error)
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))


class CodeCompileTool(BaseTool):
    name = "compile_code"
    read_only = True
    param_aliases = {"path": "file_path"}
    description = "Compiles a Python file inside the workspace to check for syntax errors. Input: 'file_path' (str)."

    def __init__(self, runtime: Optional[Any] = None):
        super().__init__(read_only=True)
        self._default_runtime_created = runtime is None
        self.runtime = runtime or RestrictedPythonRuntime()

    async def execute(self, file_path: str, **kwargs) -> ToolResult:
        try:
            ok, resolved_path, err, meta = await asyncio.to_thread(
                resolve_existing_path, file_path, self.runtime, False
            )
            if not ok:
                return ToolResult(success=False, output=meta or None, error=err)

            def _compile_sync():
                try:
                    py_compile.compile(resolved_path, doraise=True)
                    return True, None
                except py_compile.PyCompileError as ce:
                    return False, str(ce)

            success, error_msg = await asyncio.to_thread(_compile_sync)
            result = CodeCompileResult(
                file_path=resolved_path,
                success=success,
                error=error_msg,
            )
            out = _dump_model(result)
            out.update(meta)
            return ToolResult(success=success, output=out, error=error_msg)
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))
