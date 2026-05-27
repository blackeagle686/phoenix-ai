from dataclasses import dataclass, field
from typing import List, Dict, Optional

@dataclass
class FileTask:
    file_path: str
    operation: str  # create | edit | append

@dataclass
class Task:
    task_id: str
    description: str
    dependencies: List[str] = field(default_factory=list)
    tools_required: List[str] = field(default_factory=list)
    priority: str = "medium"
    status: str = "pending"

def create_task(description: str, dependencies: List[str] = None, tools_required: List[str] = None, priority: str = "medium") -> Dict:
    return {
        "description": description,
        "dependencies": dependencies or [],
        "tools_required": tools_required or [],
        "priority": priority,
        "status": "pending"
    }

def create_file_task(file_path: str, operation: str) -> Dict:
    return {
        "file_path": file_path,
        "task": operation
    }

