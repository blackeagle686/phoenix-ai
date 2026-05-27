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

class FileTask(BaseModel):
    file_path: str = Field(..., description="Path to the file to operate on")
    operation: FileOperation = Field(..., description="Operation to perform on the file")
    content: Optional[str] = Field(None, description="Content to write/append to the file")
    search_query: Optional[str] = Field(None, description="Search query for search operation")
    replace_query: Optional[str] = Field(None, description="Replacement query for search operation")

class Task(BaseModel):
    task_id: str = Field(..., description="Unique identifier for the task")
    task_summary: Optional[str] = Field(None, description="Summary of the task")
    description: str = Field(..., description="Description of the task")
    dependencies: List[str] = Field(default_factory=list, description="List of task IDs that must be completed before this task")
    tools_required: List[str] = Field(default_factory=list, description="List of tools required to complete this task")
    priority: str = Field(default="medium", description="Priority level of the task")
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="Current status of the task")
    output: Optional[str] = Field(None, description="Output of the task")
    file_tasks: List[FileTask] = Field(default_factory=list, description="List of file operations to perform")

class FileContent(BaseModel):
    file_path: str = Field(..., description="Path to the file")
    content_block: str = Field(..., description="Content to be written/appended to the file")
    from_line: int = Field(..., description="Line number to start writing/appending from")
    to_line: int = Field(..., description="Line number to end writing/appending at")
    block_summary: str = Field(..., description="Summary of the content block")

class File(BaseModel):
    file_path: str = Field(..., description="Path to the file")
    file_type: str = Field(..., description="Type of the file")
    file_id: str = Field(..., description="ID of the file")
    content: List[FileContent] = Field(default_factory=list, description="List of content blocks")
    file_summary: Optional[str] = Field(None, description="Summary of the file")
    total_lines: int = Field(..., description="Total number of lines in the file")
    total_blocks: int = Field(..., description="Total number of content blocks in the file")
    

class FileReadTask(BaseModel):
    file_path: str = Field(..., description="Path to the file to operate on")
    read_percentage: int = Field(..., description="Percentage of the file to read")
    
class FileReadResult(BaseModel):
    file_path: str = Field(..., description="Path to the file")
    content: List[FileContent] = Field(default_factory=list, description="List of content blocks")

class FileWriteTask(BaseModel):
    file_path: str = Field(..., description="Path to the file to operate on")
    from_line: int = Field(..., description="Line number to start writing/appending from")
    to_line: int = Field(..., description="Line number to end writing/appending at")
    write_content: str = Field(..., description="Content to be written/appended to the file")
    

class FileWriteResult(BaseModel):
    file_path: str = Field(..., description="Path to the file")
    content: List[FileContent] = Field(default_factory=list, description="List of content blocks")

class FileWriteReasoning(BaseModel):
    task: FileWriteTask = Field(..., description="File write task")
    result: FileWriteResult = Field(..., description="File write result")