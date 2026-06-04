from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Union, Any
from enum import Enum
from uuid import UUID

from phoenix.framework.agent.cognition.planner.schema import (
    TaskType,
    BaseFileMeta,
    FileContent,
    Task
)
from phoenix.framework.agent.cognition.reflector.schema import ReflectorInputSchema

# =========================================================================
# Base Tool Schemas
# =========================================================================

class BaseTaskInputSchema(BaseModel):
    task_id: str = Field(..., description="Task ID")
    task_description: str = Field(..., description="Task description")
    task_type: TaskType = Field(..., description="Task type")

class BaseTaskOutputSchema(BaseModel):
    task_id: str = Field(..., description="Task ID")
    success: bool = Field(..., description="Whether the task execution was successful")
    error: Optional[str] = Field(None, description="Error message if the task failed")

class BaseFileTaskInputSchema(BaseTaskInputSchema):
    file_meta: BaseFileMeta = Field(..., description="Meta information of the file")

class BaseFileTaskOutputSchema(BaseTaskOutputSchema):
    file_path: str = Field(..., description="Path to the associated file")

# =========================================================================
# Actor Core Schemas
# =========================================================================

class ActorInputSchema(BaseModel):
    """Strict input schema for the Actor module to execute a task."""
    task_id: str = Field(..., description="Unique ID of the task being executed")
    task_type: TaskType = Field(..., description="The type of the task")
    tool_name: str = Field(..., description="Name of the tool to execute")
    payload: Dict[str, Any] = Field(..., description="Arguments for the tool")

class ActorOutputSchema(BaseModel):
    """Strict output schema for the Actor module returning a task execution."""
    task_id: str = Field(..., description="Unique ID of the task that was executed")
    success: bool = Field(..., description="Whether the actor successfully completed the task")
    result: Optional[Dict[str, Any]] = Field(None, description="The normalized tool output payload")
    error_context: Optional[str] = Field(None, description="Additional actor-level error context")

class ActorToReflectorSchema(BaseModel):
    """Schema to send the Actor's tool execution output directly to the Reflector."""
    task_context: Task = Field(..., description="The full task context")
    actor_output: ActorOutputSchema = Field(..., description="The execution result of the actor")
    
    def to_reflector_input(self) -> ReflectorInputSchema:
        from phoenix.framework.agent.cognition.reflector.schema import ReflectorType, ReflectorInputSchema
        return ReflectorInputSchema(
            reflector_type=ReflectorType.TASK,
            target_id=self.task_context.task_id,
            target_content=self.actor_output.dict(),
            context=self.task_context.description
        )

# =========================================================================
# Specific Tool Schemas
# =========================================================================

class WriteTask(BaseModel):
    language: str = Field(..., description="Programming language of the file to be generated")
    description: str = Field(..., description="Detailed description of the file to be generated")
    content: str = Field(..., description="Content of the file to be generated")

#  Reading Schemas
class FileReadTask(BaseFileTaskInputSchema):
    read_percentage: int = Field(100, description="Percentage of the file to read")
    block_size: int = Field(100, description="Size of each content block to read in lines")
    from_line: Optional[int] = Field(1, description="1-indexed line number to start reading from")
    to_line: Optional[int] = Field(None, description="1-indexed line number to end reading at")
    
class FileReadResult(BaseFileTaskOutputSchema):
    content: List[FileContent] = Field(default_factory=list, description="List of content blocks")
    total_lines: int = Field(..., description="Total lines in the file")

#  Writing/Creation Schemas
class FileWriteTask(BaseFileTaskInputSchema):
    from_line: int = Field(..., description="Line number to start writing/appending from (1-indexed)")
    to_line: int = Field(..., description="Line number to end writing/appending at (1-indexed)")
    write_content: str = Field(..., description="Content to be written/appended to the file")
    
class FileWriteResult(BaseFileTaskOutputSchema):
    content: List[FileContent] = Field(default_factory=list, description="List of content blocks")
    total_lines: int = Field(..., description="Total number of lines in the file after write")

class FileWriteReasoning(BaseModel):
    task: FileWriteTask = Field(..., description="File write task details")
    result: FileWriteResult = Field(..., description="File write result details")

#  Searching Schemas
class FileSearchMatch(BaseModel):
    line_number: int = Field(..., description="1-indexed line number of the match")
    line_content: str = Field(..., description="The content of the matching line")
    block_index: Optional[int] = Field(None, description="Index of the FileContent block containing this match")

class FileSearchTask(BaseFileTaskInputSchema):
    search_query: str = Field(..., description="Query string or pattern to look for")
    is_regex: bool = Field(default=False, description="Treat query as a regex pattern")
    case_sensitive: bool = Field(default=False, description="Perform case-sensitive search")

class FileSearchResult(BaseFileTaskOutputSchema):
    matches: List[FileSearchMatch] = Field(default_factory=list, description="List of search matches")
    total_matches: int = Field(..., description="Total matches found in the file")

#  Updating/Editing Schemas
class ReplacementChunk(BaseModel):
    from_line: int = Field(..., description="1-indexed start line of the range to replace")
    to_line: int = Field(..., description="1-indexed end line of the range to replace (inclusive)")
    target_content: str = Field(..., description="The exact content targeted for replacement")
    replacement_content: str = Field(..., description="The content to substitute into the file")

class FileUpdateTask(BaseFileTaskInputSchema):
    chunks: List[ReplacementChunk] = Field(..., description="List of replacement chunks for the file")

class FileUpdateResult(BaseFileTaskOutputSchema):
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

class WebSearchTask(BaseTaskInputSchema):
    query: str = Field(..., description="Search engine query string")

class WebSearchResult(BaseTaskOutputSchema):
    query: str = Field(..., description="Search query performed")
    results: List[WebSearchItem] = Field(default_factory=list, description="List of search result items")

# Code Execution (REPL) Schemas
class CodeExecutionTask(BaseTaskInputSchema):
    code: str = Field(..., description="Safe python code to execute")

class CodeExecutionResult(BaseTaskOutputSchema):
    output: str = Field(..., description="Output from stdout or evaluation result")

# Python Analysis Schemas
class PythonMethodInfo(BaseModel):
    name: str = Field(..., description="Name of the method/function")
    line_start: int = Field(..., description="Start line number in source")
    line_end: int = Field(..., description="End line number in source")

class PythonClassInfo(BaseModel):
    line_start: int = Field(..., description="Start line number in source")
    line_end: int = Field(..., description="End line number in source")
    methods: List[PythonMethodInfo] = Field(default_factory=list, description="List of methods defined in the class")

class PythonAnalysisTask(BaseFileTaskInputSchema):
    pass

class PythonAnalysisResult(BaseFileTaskOutputSchema):
    classes: Dict[str, PythonClassInfo] = Field(default_factory=dict, description="Dictionary mapping class names to class metadata")
    functions: List[PythonMethodInfo] = Field(default_factory=list, description="Top-level functions defined in the file")
    imports: List[str] = Field(default_factory=list, description="List of import statement strings in the file")

# Multi-Block Update Schemas
class MultiBlockUpdateEdit(BaseModel):
    target: str = Field(..., description="Target substring to replace")
    replacement: str = Field(..., description="Replacement substring")

class MultiBlockUpdateTask(BaseFileTaskInputSchema):
    edits: List[MultiBlockUpdateEdit] = Field(..., description="List of search-and-replace pairs")

class MultiBlockUpdateResult(BaseFileTaskOutputSchema):
    applied_count: int = Field(..., description="Number of edits successfully applied")
    output: str = Field(..., description="Status summary or error details")

# Command / Shell Execution Schemas
class CommandExecutionTask(BaseTaskInputSchema):
    command: str = Field(..., description="Shell/Terminal command to execute")
    cwd: Optional[str] = Field(None, description="Working directory context for execution")

class CommandExecutionResult(BaseTaskOutputSchema):
    command: str = Field(..., description="Executed command")
    stdout: str = Field(..., description="Stdout stream output")
    stderr: str = Field(..., description="Stderr stream output")
    exit_code: int = Field(..., description="Exit code returned by shell execution")

# Code Compilation / Syntax check Schemas
class CodeCompileTask(BaseFileTaskInputSchema):
    pass

class CodeCompileResult(BaseFileTaskOutputSchema):
    pass
