import ast
import io
import sys
import asyncio
from typing import Optional
from .base import BaseRuntime, ExecutionResult

class RestrictedPythonRuntime(BaseRuntime):
    """
    Zero-dependency runtime that uses AST filtering to block dangerous Python modules.
    For shell commands, it provides a basic subprocess executor.
    """

    DANGEROUS_MODULES = {
        "os", "sys", "subprocess", "shutil", "pty", "socket", "builtins", 
        "pty", "macpath", "ntpath", "posixpath", "cmd", "shlex"
    }

    def _is_safe_code(self, code: str) -> tuple[bool, Optional[str]]:
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, f"Syntax Error: {str(e)}"

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    base_module = alias.name.split('.')[0]
                    if base_module in self.DANGEROUS_MODULES:
                        return False, f"Importing '{base_module}' is blocked for security reasons."
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    base_module = node.module.split('.')[0]
                    if base_module in self.DANGEROUS_MODULES:
                        return False, f"Importing from '{base_module}' is blocked for security reasons."
            # Additionally, block calls to `eval` and `exec` inside the code
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ("eval", "exec", "open"):
                        return False, f"Function '{node.func.id}' is blocked for security reasons."
        
        return True, None

    async def execute_code(self, code: str, language: str = "python") -> ExecutionResult:
        if language.lower() != "python":
            return ExecutionResult(success=False, output="", error=f"Unsupported language: {language}")

        is_safe, error_msg = self._is_safe_code(code)
        if not is_safe:
            return ExecutionResult(success=False, output="", error=f"Security Violation: {error_msg}", exit_code=1)

        local_vars = {}
        old_stdout = sys.stdout
        redirected_output = sys.stdout = io.StringIO()
        
        try:
            # We explicitly define __builtins__ to prevent access to the host's full builtins
            # but allow standard operations. Note: 'open', 'eval', 'exec' are already blocked by AST,
            # but we can restrict further if needed. We'll pass standard builtins for now.
            exec(code, {"__builtins__": __builtins__}, local_vars)
            success = True
            error_msg = None
        except Exception as e:
            sys.stdout = old_stdout
            return ExecutionResult(success=False, output="", error=f"Execution error: {str(e)}", exit_code=1)
            
        sys.stdout = old_stdout
        output = redirected_output.getvalue()
        
        if not output and "result" in local_vars:
            output = str(local_vars["result"])
            
        return ExecutionResult(
            success=success,
            output=output.strip() if output else "Executed successfully (no output)",
            error=error_msg,
            exit_code=0
        )

    async def execute_command(self, command: str, cwd: Optional[str] = None) -> ExecutionResult:
        """
        Executes a shell command. 
        Note: Full native sandboxing (like blocking file access) requires OS-level jails (Bubblewrap) or Docker.
        Here we provide a restricted basic execution.
        """
        # Basic hardcoded heuristics for extreme commands
        dangerous_patterns = ["rm -rf /", "mkfs", ":(){ :|:& };:"]
        for pattern in dangerous_patterns:
            if pattern in command:
                return ExecutionResult(
                    success=False, 
                    output="", 
                    error=f"Security Violation: Command contains dangerous pattern.", 
                    exit_code=1
                )

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
            
            return ExecutionResult(
                success=success,
                output=stdout_str,
                error=stderr_str if not success else None,
                exit_code=exit_code
            )
        except Exception as e:
            return ExecutionResult(success=False, output="", error=str(e), exit_code=1)
