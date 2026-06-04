from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Union
from enum import Enum
from uuid import UUID

class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    SKIPPED = "skipped"

class TaskType(str, Enum): 
    

class FileOperation(str, Enum):
    CREATE = "create"
    EDIT = "edit"
    APPEND = "append"
    READ = "read"
    DELETE = "delete"
    SEARCH = "search"
    UPDATE = "update"

class FileTask(BaseModel):
    file_path: str = Field(..., description="Path to the file to operate on")
    operation: FileOperation = Field(..., description="Operation to perform on the file")
    content: Optional[str] = Field(None, description="Content to write/append to the file")
    search_query: Optional[str] = Field(None, description="Search query for search operation")
    replace_query: Optional[str] = Field(None, description="Replacement query for search operation")

class Prompt(BaseModel):
    id: UUID = Field(default_factory=uuid.uuid4, description="Unique identifier for the user prompt")
    user_id: UUID = Field(default_factory=uuid.uuid4, description="Unique identifier for the user")
    project_id: UUID = Field(default_factory=uuid.uuid4, description="Unique identifier for the project")
    user_message: str = Field(..., description="User message")
    system_message: str = Field(..., description="System message")
    tokens_length: int = Field(..., description="Tokens length")

class Task(BaseModel):
    prompt_id: UUID = Field(default_factory=uuid.uuid4, description="Unique identifier for the prompt")
    task_id: str = Field(..., description="Unique identifier for the task")
    task_type: str = Field(..., description="Type of the task")
    task_title: str = Field(..., description="Title of the task")
    description: str = Field(..., description="Description of the task")
    dependencies: List[str] = Field(default_factory=list, description="List of task IDs that must be completed before this task")
    priority: str = Field(default="medium", description="Priority level of the task")
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="Current status of the task")
    task_summary: str = Field(None, description="Summary of the task")



class FileContent(BaseModel):
    file_path: str = Field(..., description="Path to the file")
    content_block: str = Field(..., description="Content of this specific block")
    from_line: int = Field(..., description="1-indexed line number where this block starts")
    to_line: int = Field(..., description="1-indexed line number where this block ends (inclusive)")
    block_summary: Optional[str] = Field(None, description="Summary of the content block")

class File(BaseModel):
    file_path: str = Field(..., description="Path to the file")
    file_type: str = Field(..., description="Type/extension of the file")
    file_id: Optional[str] = Field(None, description="Unique ID of the file")
    content: List[FileContent] = Field(default_factory=list, description="List of content blocks")
    file_summary: Optional[str] = Field(None, description="Summary of the file")
    total_lines: int = Field(..., description="Total number of lines in the file")
    total_blocks: int = Field(..., description="Total number of content blocks in the file")

#  Reading Schemas

class FileReadTask(BaseModel):
    file_path: str = Field(..., description="Path to the file to operate on") # llm define
    read_percentage: int = Field(100, description="Percentage of the file to read") # system define 
    block_size: int = Field(100, description="Size of each content block to read in lines") # system define 
    from_line: Optional[int] = Field(1, description="1-indexed line number to start reading from") # llm define
    to_line: Optional[int] = Field(None, description="1-indexed line number to end reading at") # llm define
    
class FileReadResult(BaseModel):
    file_path: str = Field(..., description="Path to the file") # llm define
    content: List[FileContent] = Field(default_factory=list, description="List of content blocks") # llm define
    total_lines: int = Field(..., description="Total lines in the file") # system define

#  Writing/Creation Schemas

class FileWriteTask(BaseModel):
    file_path: str = Field(..., description="Path to the file to operate on") # llm define
    from_line: int = Field(..., description="Line number to start writing/appending from (1-indexed)") # llm define
    to_line: int = Field(..., description="Line number to end writing/appending at (1-indexed)") # llm define
    write_content: str = Field(..., description="Content to be written/appended to the file") # llm define
    
class FileWriteResult(BaseModel):
    file_path: str = Field(..., description="Path to the file") # llm define
    content: List[FileContent] = Field(default_factory=list, description="List of content blocks") # llm define
    total_lines: int = Field(..., description="Total number of lines in the file after write") # system define

class FileWriteReasoning(BaseModel):
    task: FileWriteTask = Field(..., description="File write task details") # llm define
    result: FileWriteResult = Field(..., description="File write result details") # llm define

#  Searching Schemas

class FileSearchMatch(BaseModel):
    line_number: int = Field(..., description="1-indexed line number of the match")
    line_content: str = Field(..., description="The content of the matching line")
    block_index: Optional[int] = Field(None, description="Index of the FileContent block containing this match")

class FileSearchTask(BaseModel):
    file_path: str = Field(..., description="Path to the file to search within")
    search_query: str = Field(..., description="Query string or pattern to look for")
    is_regex: bool = Field(default=False, description="Treat query as a regex pattern")
    case_sensitive: bool = Field(default=False, description="Perform case-sensitive search")

class FileSearchResult(BaseModel):
    file_path: str = Field(..., description="Path to the searched file")
    matches: List[FileSearchMatch] = Field(default_factory=list, description="List of search matches")
    total_matches: int = Field(..., description="Total matches found in the file")

#  Updating/Editing Schemas

class ReplacementChunk(BaseModel):
    from_line: int = Field(..., description="1-indexed start line of the range to replace")
    to_line: int = Field(..., description="1-indexed end line of the range to replace (inclusive)")
    target_content: str = Field(..., description="The exact content targeted for replacement")
    replacement_content: str = Field(..., description="The content to substitute into the file")

class FileUpdateTask(BaseModel):
    file_path: str = Field(..., description="Path to the file to update/edit")
    chunks: List[ReplacementChunk] = Field(..., description="List of replacement chunks for the file")

class FileUpdateResult(BaseModel):
    file_path: str = Field(..., description="Path to the file")
    success: bool = Field(..., description="Whether the update operation was fully successful")
    content: List[FileContent] = Field(default_factory=list, description="List of updated content blocks")
    total_lines: int = Field(..., description="Total lines in the file after update")

class FileUpdateReasoning(BaseModel):
    task: FileUpdateTask = Field(..., description="File update task details")
    reasoning: Optional[str] = Field(None, description="Explanation or step-by-step reasoning for the update")
    result: FileUpdateResult = Field(..., description="File update result details")

# Web Search Schemas
class WebSearchItem(BaseModel):
    title: str = Field(..., description="Title of the search result")
    snippet: str = Field(..., description="Snippet/description of the search result")
    url: str = Field(..., description="URL of the search result")

class WebSearchTask(BaseModel):
    query: str = Field(..., description="Search engine query string")

class WebSearchResult(BaseModel):
    query: str = Field(..., description="Search query performed")
    results: List[WebSearchItem] = Field(default_factory=list, description="List of search result items")

# Code Execution (REPL) Schemas
class CodeExecutionTask(BaseModel):
    code: str = Field(..., description="Safe python code to execute")

class CodeExecutionResult(BaseModel):
    success: bool = Field(..., description="Whether code ran without raising unhandled exceptions")
    output: str = Field(..., description="Output from stdout or evaluation result")
    error: Optional[str] = Field(None, description="Exception stack trace if code failed")

# Python Analysis Schemas
class PythonMethodInfo(BaseModel):
    name: str = Field(..., description="Name of the method/function")
    line_start: int = Field(..., description="Start line number in source")
    line_end: int = Field(..., description="End line number in source")

class PythonClassInfo(BaseModel):
    line_start: int = Field(..., description="Start line number in source")
    line_end: int = Field(..., description="End line number in source")
    methods: List[PythonMethodInfo] = Field(default_factory=list, description="List of methods defined in the class")

class PythonAnalysisTask(BaseModel):
    file_path: str = Field(..., description="Path to the Python file to analyze")

class PythonAnalysisResult(BaseModel):
    file_path: str = Field(..., description="Path of analyzed file")
    classes: Dict[str, PythonClassInfo] = Field(default_factory=dict, description="Dictionary mapping class names to class metadata")
    functions: List[PythonMethodInfo] = Field(default_factory=list, description="Top-level functions defined in the file")
    imports: List[str] = Field(default_factory=list, description="List of import statement strings in the file")

# Multi-Block Update Schemas
class MultiBlockUpdateEdit(BaseModel):
    target: str = Field(..., description="Target substring to replace")
    replacement: str = Field(..., description="Replacement substring")

class MultiBlockUpdateTask(BaseModel):
    file_path: str = Field(..., description="Path to the file to modify")
    edits: List[MultiBlockUpdateEdit] = Field(..., description="List of search-and-replace pairs")

class MultiBlockUpdateResult(BaseModel):
    file_path: str = Field(..., description="Path to the modified file")
    success: bool = Field(..., description="Whether all edits were successfully applied")
    applied_count: int = Field(..., description="Number of edits successfully applied")
    output: str = Field(..., description="Status summary or error details")

# Command / Shell Execution Schemas
class CommandExecutionTask(BaseModel):
    command: str = Field(..., description="Shell/Terminal command to execute")
    cwd: Optional[str] = Field(None, description="Working directory context for execution")

class CommandExecutionResult(BaseModel):
    command: str = Field(..., description="Executed command")
    success: bool = Field(..., description="Whether exit code was 0")
    stdout: str = Field(..., description="Stdout stream output")
    stderr: str = Field(..., description="Stderr stream output")
    exit_code: int = Field(..., description="Exit code returned by shell execution")

# Code Compilation / Syntax check Schemas
class CodeCompileTask(BaseModel):
    file_path: str = Field(..., description="Path of Python/source file to check/compile")

class CodeCompileResult(BaseModel):
    file_path: str = Field(..., description="Path of checked file")
    success: bool = Field(..., description="Whether file compiles/checks with no errors")
    error: Optional[str] = Field(None, description="Compilation or syntax error details if failed")





