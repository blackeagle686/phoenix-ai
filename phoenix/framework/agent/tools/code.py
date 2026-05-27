from phoenix.framework.agent.tools.base import BaseTool, ToolResult
import ast
import os
import json
import io
import sys
import asyncio
import py_compile
from typing import Optional
from phoenix.framework.agent.cognition.planner.schema import (
    CodeExecutionResult,
    PythonAnalysisResult,
    PythonClassInfo,
    PythonMethodInfo,
    CommandExecutionResult,
    CodeCompileResult
)

class CodeExecutionTool(BaseTool):
    name = "python_repl"
    description = "Executes safe python code to perform calculations or logic. Input: 'code' (str)."

    async def execute(self, code: str, **kwargs) -> ToolResult:
        try:
            local_vars = {}
            old_stdout = sys.stdout
            redirected_output = sys.stdout = io.StringIO()
            
            try:
                exec(code, {"__builtins__": __builtins__}, local_vars)
                success = True
                error_msg = None
            except Exception as e:
                sys.stdout = old_stdout
                return ToolResult(
                    success=False,
                    output=CodeExecutionResult(success=False, output="", error=str(e)).dict(),
                    error=f"Execution error: {str(e)}"
                )
                
            sys.stdout = old_stdout
            output = redirected_output.getvalue()
            
            if not output and "result" in local_vars:
                output = str(local_vars["result"])
                
            result = CodeExecutionResult(
                success=success,
                output=output.strip() if output else "Executed successfully (no output)",
                error=error_msg
            )
            return ToolResult(success=True, output=result.dict())
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))

class PythonAnalyzerTool(BaseTool):
    """
    Fast AST-based analyzer for Python files to map classes and functions.
    """
    name = "python_analyzer"
    description = "Analyzes a Python file and returns an index of all classes and functions with their line numbers. Input: 'file_path' (str)."

    async def execute(self, file_path: str, **kwargs) -> ToolResult:
        try:
            if not os.path.exists(file_path):
                return ToolResult(success=False, output=None, error=f"File not found: {file_path}")

            with open(file_path, "r", encoding="utf-8") as f:
                source = f.read()

            tree = ast.parse(source)
            classes_map = {}
            functions_list = []
            imports_list = []

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    methods = []
                    for n in node.body:
                        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            methods.append(PythonMethodInfo(
                                name=n.name,
                                line_start=n.lineno,
                                line_end=getattr(n, "end_lineno", n.lineno)
                            ))
                    classes_map[node.name] = PythonClassInfo(
                        line_start=node.lineno,
                        line_end=getattr(node, "end_lineno", node.lineno),
                        methods=methods
                    )

            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    functions_list.append(PythonMethodInfo(
                        name=node.name,
                        line_start=node.lineno,
                        line_end=getattr(node, "end_lineno", node.lineno)
                    ))
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    imports_list.append(ast.unparse(node))

            result = PythonAnalysisResult(
                file_path=file_path,
                classes=classes_map,
                functions=functions_list,
                imports=imports_list
            )
            return ToolResult(success=True, output=result.dict())

        except SyntaxError as e:
            return ToolResult(success=False, output=None, error=f"Syntax error in Python file: {e}")
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))

class CommandExecutionTool(BaseTool):
    name = "execute_command"
    description = "Executes a shell/terminal command in the specified directory and returns stdout, stderr, and exit code. Input: 'command' (str), 'cwd' (Optional[str])."

    async def execute(self, command: str, cwd: Optional[str] = None, **kwargs) -> ToolResult:
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd
            )
            stdout, stderr = await proc.communicate()
            stdout_str = stdout.decode('utf-8', errors='replace')
            stderr_str = stderr.decode('utf-8', errors='replace')
            exit_code = proc.returncode
            success = (exit_code == 0)
            
            result = CommandExecutionResult(
                command=command,
                success=success,
                stdout=stdout_str,
                stderr=stderr_str,
                exit_code=exit_code
            )
            return ToolResult(success=success, output=result.dict(), error=stderr_str if not success else None)
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))

class CodeCompileTool(BaseTool):
    name = "compile_code"
    description = "Compiles a Python code file to check for syntax and compilation errors. Input: 'file_path' (str)."

    async def execute(self, file_path: str, **kwargs) -> ToolResult:
        try:
            if not os.path.exists(file_path):
                return ToolResult(success=False, output=None, error=f"File not found: {file_path}")
            
            try:
                py_compile.compile(file_path, doraise=True)
                success = True
                error_msg = None
            except py_compile.PyCompileError as e:
                success = False
                error_msg = str(e)
            
            result = CodeCompileResult(
                file_path=file_path,
                success=success,
                error=error_msg
            )
            return ToolResult(success=success, output=result.dict(), error=error_msg)
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))
