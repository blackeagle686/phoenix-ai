from pydantic import BaseModel, Field
from typing import Dict, List
from enum import Enum

class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    SKIPPED = "skipped"

class FileOperation(str, Enum):
    CREATE = "create"
    EDIT = "edit"
    APPEND = "append"
    READ = "read"
    DELETE = "delete"
    SEARCH = "search"
    