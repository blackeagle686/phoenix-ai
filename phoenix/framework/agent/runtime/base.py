from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

@dataclass
class ExecutionResult:
    success: bool
    output: str
    error: Optional[str] = None
    exit_code: int = 0

class BaseRuntime(ABC):
    """
    Abstract Base Class for an isolated runtime environment.
    Provides methods to safely execute code and shell commands.
    """
    
    @abstractmethod
    async def execute_code(self, code: str, language: str = "python") -> ExecutionResult:
        """
        Executes a snippet of code (e.g. Python, JS) in a sandbox.
        """
        pass

    @abstractmethod
    async def execute_command(self, command: str, cwd: Optional[str] = None) -> ExecutionResult:
        """
        Executes a shell command or bash script in an isolated environment.
        """
        pass

    @abstractmethod
    async def execute_io(self, operation: str, file_path: str, content: Optional[str] = None) -> ExecutionResult:
        """
        Executes a strict File I/O operation (create, read, edit, delete).
        """
        pass
