from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Union, Any
from enum import Enum
from uuid import UUID, uuid4
from datetime import datetime
from phoenix.framework.agent.cognition.utils.id import generate_unique_id

# =========================================================================
# ENUMS
# =========================================================================
class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"

class TaskType(str, Enum):
    READ = "read"
    WRITE = "write"
    SEARCH = "search"
    UPDATE = "update"
    DELETE = "delete"
    OTHER = "other"
    # Keeping it simple, can add others if needed

class TaskPriority(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class FileOperation(str, Enum):
    CREATE = "create"
    CREATE_DIR = "create_dir"
    READ = "read"
    WRITE = "write"
    APPEND = "append"
    SEARCH = "search"
    REPLACE = "replace"
    DELETE = "delete"
    CHMOD = "chmod"

class SolutionType(str, Enum): 
    PLAN = "plan"
    CODE = "code"
    TERMINAL = "terminal"
    NETWORK = "network"
    MISSION = "mission"
    FASTANSWER = "fastanswer"
    OTHER = "other"

class ProblemComplexity(str, Enum): 
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"

class ReflectorType(str, Enum):
    TASK = "task"
    PROBLEM = "problem"
    SOLUTION = "solution"

class FileUpdateStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"

# =========================================================================
# BASE UTILS & META
# =========================================================================

class BaseReflectorMeta(BaseModel):
    rating: int = Field(..., ge=1, le=10, description="Evaluation rating on a scale of 1 to 10")
    feedback: str = Field(..., description="Constructive feedback or critique")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score of this evaluation between 0.0 and 1.0")
    reasoning: str = Field(..., description="Detailed reasoning for the assigned rating and feedback")

class ReflectorInputSchema(BaseModel):
    reflector_type: ReflectorType = Field(..., description="The type of the item being evaluated")
    target_id: str = Field(..., description="The unique identifier of the target item being evaluated")
    target_content: Any = Field(..., description="The serialized content of the item being evaluated")
    context: Optional[str] = Field(None, description="Overarching objective or context to guide the evaluation")

class ReflectorOutputSchema(BaseReflectorMeta):
    pass

class FileContent(BaseModel):
    file_path: str = Field(..., description="Path to the file")
    content_block: str = Field(..., description="Raw text or code content of this specific block")
    from_line: int = Field(..., description="1-indexed line number where this block starts")
    to_line: int = Field(..., description="1-indexed line number where this block ends (inclusive)")
    token_count: Optional[int] = Field(None, description="Calculated LLM token count for this specific block")
    overlap_lines: int = Field(default=0, description="Number of lines duplicated from the previous block for context preservation")
    block_summary: Optional[str] = Field(None, description="AI-generated summary of this content block")
    vector_id: Optional[str] = Field(None, description="Reference ID if this block is embedded in a Vector Database")

class BaseFileMeta(BaseModel):
    file_path: str = Field(..., description="Path of the file")
    file_name: str = Field(..., description="Name of the file")
    lines_count: int = Field(..., description="Number of lines in the file")
    file_size: int = Field(..., description="Size of the file in bytes")
    file_type: str = Field(..., description="Type of the file")
    last_modified_time: int = Field(..., description="Last modified time of the file in seconds since epoch")

class FileIOParams(BaseModel):
    file_path: str = Field(..., description="ABSOLUTE full path to the target file")
    operation: FileOperation = Field(..., description="The specific file sub-operation")
    content: Optional[Union[str, bytes]] = Field(None, description="Raw text or binary bytes to write")
    encoding: str = Field(default="utf-8", description="File text encoding style")
    offset: Optional[int] = Field(None, description="Byte position to seek")
    length: Optional[int] = Field(None, description="Number of bytes to read")
    search_query: Optional[str] = Field(None, description="Regex or text token to look up inside the file")
    replace_query: Optional[str] = Field(None, description="The content used to substitute the search match")
    permissions: Optional[str] = Field(None, description="Linux octal permission string")

class File(BaseModel):
    file_id: str = Field(..., description="Unique deterministic UUID or hash of the file path")
    file_path: str = Field(..., description="Absolute or relative path to the file")
    file_type: str = Field(..., description="File extension or MIME type")
    content: List[FileContent] = Field(default_factory=list, description="Ordered list of segmented content blocks")
    total_lines: int = Field(..., description="Total line count of the source file")
    total_blocks: int = Field(..., description="Total number of split blocks")
    total_tokens: Optional[int] = Field(None, description="Aggregate token count of the entire file")
    file_hash: Optional[str] = Field(None, description="MD5/SHA256 checksum to detect external modifications")
    encoding: str = Field(default="utf-8", description="File text encoding")
    file_summary: Optional[str] = Field(None, description="High-level systemic summary")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Custom extensions")

    class Config:
        populate_by_name = True

class FileIOMeta(BaseModel):
    file_meta: BaseFileMeta = Field(..., description="Configuration of the file")
    from_line: int = Field(..., description="Start line number of the file")
    to_line: int = Field(..., description="End line number of the file")
    status: FileUpdateStatus = Field(..., description="Status of the file update")

# =========================================================================
# OLD PLANNER/TASK SCHEMAS
# =========================================================================

class Solution(BaseModel):
    id: UUID = Field(default_factory=uuid4, description="ID")
    description: str = Field(..., description="solution description")
    solution_type: SolutionType = Field(..., description="Type of solution")
    content: str = Field(..., description="solution content")
    reflector_result: BaseReflectorMeta = Field(..., description="reflector result")

class Problem(BaseModel): 
    id: UUID = Field(default_factory=uuid4, description="ID") 
    description: str = Field(..., description="problem description")
    solution: List[Solution] = Field(..., description="solution")
    best_solution: Solution = Field(..., description="best solution")
    complexity: ProblemComplexity = Field(..., description="complexity")
    reflector_result: BaseReflectorMeta = Field(..., description="reflector result")

class Task(BaseModel):
    prompt_id: UUID = Field(default_factory=uuid4, description="Unique identifier for the parent session/prompt request")
    task_id: str = Field(default_factory=lambda: uuid4().hex, description="Unique deterministic identifier for this specific task")
    dependencies: List[str] = Field(default_factory=list, description="List of task_ids that must complete successfully first")
    task_type: TaskType = Field(..., description="The specific systemic I/O operation archetype")
    priority: TaskPriority = Field(default=TaskPriority.MEDIUM, description="The execution urgency tier")
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="Current workflow state machine status")
    task_title: str = Field(..., description="Short title of the task")
    description: str = Field(..., description="Verbose description detailing task goals")
    task_summary: Optional[str] = Field(None, description="Post-execution summary populated after completion")
    complexity: ProblemComplexity = Field(..., description="complexity")
    problems: List[Problem] = Field(..., description="problems")
    repeat_count: int = Field(1, description="Number of times the task should be repeated")
    payload: Dict[str, Any] = Field(default_factory=dict, description="Input parameters required by the driver")
    result: Optional[Dict[str, Any]] = Field(None, description="Output returned by the executing module/driver")
    error: Optional[str] = Field(None, description="Error tracking message if status shifts to FAILED")
    created_by: str = Field(..., description="The identifier of the agent/orchestrator that generated this task")
    assigned_to: Optional[str] = Field(None, description="The targeted agent, execution worker pool, or hardware driver")
    timeout: float = Field(default=30.0, description="Max execution time window in seconds before the task is forcefully killed")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Timestamp when the task was initially queued")
    executed_at: Optional[datetime] = Field(None, description="Timestamp when the execution worker actually started processing")
    reflector_result: BaseReflectorMeta = Field(..., description="reflector result")

    class Config:
        use_enum_values = True

class Prompt(BaseModel):
    id: UUID = Field(default_factory=uuid4, description="Unique identifier for the user prompt")
    user_id: Optional[UUID] = Field(None, description="Unique identifier for the user")
    project_id: Optional[UUID] = Field(None, description="Unique identifier for the project")
    user_message: str = Field(..., description="User message")
    system_message: str = Field(..., description="System message")
    tokens_length: int = Field(..., description="Tokens length")

class PlannerInputSchema(BaseModel):
    prompt: Prompt = Field(..., description="The user prompt and session details")
    context: Optional[str] = Field(None, description="Additional context or memory for the planner")
    existing_tasks: List[Task] = Field(default_factory=list, description="Current state of existing tasks")
    previous_results: Optional[str] = Field(None, description="Results from previous actions or executions")

class PlannerOutputSchema(BaseModel):
    plan_id: UUID = Field(default_factory=uuid4, description="Unique ID for this planner interaction")
    response: str = Field(..., description="Conversational response or direct answer to the user")
    problems: List[Problem] = Field(default_factory=list, description="Identified problems and complexities")
    solutions: List[Solution] = Field(default_factory=list, description="Direct solutions, code snippets, or fast answers provided")
    tasks: List[Task] = Field(default_factory=list, description="The ordered sequence of actionable tasks to execute if needed")
    summary: str = Field(..., description="A high-level summary of the planner's reasoning and output")

# =========================================================================
# NEW BRAIN SCHEMAS
# =========================================================================

class ToolCall(BaseModel):
    tool_name: str = Field(..., description="Name of the tool to execute")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="Arguments to pass to the tool")

class TaskExecutionSchema(BaseModel):
    thought_process: str = Field(..., description="Detailed analysis of the problem and the step-by-step approach to solve it")
    tools_to_call: List[ToolCall] = Field(default_factory=list, description="Specific tool calls required to enact the solution")

class ReflectionSchema(BaseModel):
    status: TaskStatus = Field(..., description="The evaluation status: 'done', 'failed', or 'in_progress'")
    feedback: str = Field(..., description="Detailed feedback on the runtime output and what needs to happen next")
    rating: int = Field(..., ge=1, le=10, description="Rating of the execution from 1 to 10")
    is_task_complete: bool = Field(..., description="Whether the entire task is now fully complete")

class TaskSchema(BaseModel):
    task_id: str = Field(default_factory=generate_unique_id, description="Unique ID for the task")
    task_type: TaskType = Field(default=TaskType.OTHER, description="The operational archetype of the task")
    priority: TaskPriority = Field(default=TaskPriority.MEDIUM, description="Execution priority")
    description: str = Field(..., description="What the task requires")
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="Task workflow status")

class PlanSchema(BaseModel):
    objective: str = Field(..., description="The main overall objective")
    tasks: List[TaskSchema] = Field(default_factory=list, description="List of ordered tasks to complete the objective")

# =========================================================================
# ACTOR SCHEMAS
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
    tool_name: str = Field(default="unknown", description="Name of the tool that was executed")
    success: bool = Field(..., description="Whether the actor successfully completed the task")
    result: Optional[Dict[str, Any]] = Field(None, description="The normalized tool output payload")
    error_context: Optional[str] = Field(None, description="Additional actor-level error context")
    reflection: Optional[Any] = Field(None, description="Optional reflection output from the Reflector module")

class ActorToReflectorSchema(BaseModel):
    """Schema to send the Actor's tool execution output directly to the Reflector."""
    task_context: TaskSchema = Field(..., description="The full task context")
    actor_output: ActorOutputSchema = Field(..., description="The execution result of the actor")

# Base Tool Schemas
class BaseTaskInputSchema(BaseModel):
    pass
class BaseTaskOutputSchema(BaseModel):
    pass
class BaseFileTaskInputSchema(BaseTaskInputSchema):
    file_meta: BaseFileMeta = Field(..., description="Meta information of the file")
class BaseFileTaskOutputSchema(BaseTaskOutputSchema):
    file_path: str = Field(..., description="Path to the associated file")

class WebSearchItem(BaseModel):
    title: str = Field(..., description="Title of the search result")
    snippet: str = Field(..., description="Snippet/description of the search result")
    url: str = Field(..., description="URL of the search result")
class WebSearchResult(BaseTaskOutputSchema):
    query: str = Field(..., description="Search query performed")
    results: List[WebSearchItem] = Field(default_factory=list, description="List of search result items")
class MultiBlockUpdateEdit(BaseModel):
    target: str = Field(..., description="Target substring to replace")
    replacement: str = Field(..., description="Replacement substring")
class MultiBlockUpdateResult(BaseFileTaskOutputSchema):
    applied_count: int = Field(..., description="Number of edits successfully applied")
    output: str = Field(..., description="Status summary or error details")

# Adding remaining specific tool schemas to ensure compatibility
class FileReadTask(BaseFileTaskInputSchema):
    read_percentage: int = Field(100)
    block_size: int = Field(100)
    from_line: Optional[int] = Field(1)
    to_line: Optional[int] = Field(None)
    
class FileReadResult(BaseFileTaskOutputSchema):
    content: List[FileContent] = Field(default_factory=list)
    total_lines: int = Field(...)

class FileWriteTask(BaseFileTaskInputSchema):
    from_line: int = Field(...)
    to_line: int = Field(...)
    write_content: str = Field(...)
    
class FileWriteResult(BaseFileTaskOutputSchema):
    content: List[FileContent] = Field(default_factory=list)
    total_lines: int = Field(...)

class ReplacementChunk(BaseModel):
    from_line: int = Field(...)
    to_line: int = Field(...)
    target_content: Optional[str] = Field(None)
    replacement_content: str = Field(...)

class FileUpdateTask(BaseFileTaskInputSchema):
    chunks: List[ReplacementChunk] = Field(...)

class FileUpdateResult(BaseFileTaskOutputSchema):
    content: List[FileContent] = Field(default_factory=list)
    total_lines: int = Field(...)

class CommandExecutionTask(BaseTaskInputSchema):
    command: str = Field(...)
    cwd: Optional[str] = Field(None)

class CommandExecutionResult(BaseTaskOutputSchema):
    command: str = Field(...)
    stdout: str = Field(...)
    stderr: str = Field(...)
    exit_code: int = Field(...)

class CodeExecutionTask(BaseTaskInputSchema):
    code: str = Field(..., description="Safe python code to execute")

class CodeExecutionResult(BaseTaskOutputSchema):
    output: str = Field(..., description="Output from stdout or evaluation result")

class CodeCompileTask(BaseFileTaskInputSchema):
    pass

class CodeCompileResult(BaseFileTaskOutputSchema):
    pass

class PythonMethodInfo(BaseModel):
    name: str = Field(...)
    line_start: int = Field(...)
    line_end: int = Field(...)

class PythonClassInfo(BaseModel):
    line_start: int = Field(...)
    line_end: int = Field(...)
    methods: List[PythonMethodInfo] = Field(default_factory=list)

class PythonAnalysisTask(BaseFileTaskInputSchema):
    pass

class PythonAnalysisResult(BaseFileTaskOutputSchema):
    classes: Dict[str, PythonClassInfo] = Field(default_factory=dict)
    functions: List[PythonMethodInfo] = Field(default_factory=list)
    imports: List[str] = Field(default_factory=list)

class FileSearchMatch(BaseModel):
    line_number: int = Field(...)
    line_content: str = Field(...)
    block_index: Optional[int] = Field(None)

class FileSearchTask(BaseFileTaskInputSchema):
    search_query: str = Field(...)
    is_regex: bool = Field(default=False)
    case_sensitive: bool = Field(default=False)

class FileSearchResult(BaseFileTaskOutputSchema):
    matches: List[FileSearchMatch] = Field(default_factory=list)
    total_matches: int = Field(...)
