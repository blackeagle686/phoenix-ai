import json
from pydantic import BaseModel, Field

try:
    from pydantic import model_validator
except ImportError:
    model_validator = None

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

_TASK_TYPE_ALIASES: Dict[str, str] = {
    "analyze": "read", "analyse": "read", "inspect": "read", "audit": "read",
    "verify": "read", "check": "read", "review": "read", "test": "read", "view": "read", "explain": "read",
    "create": "write", "new": "write", "add": "write", "implement": "write", "generate": "write", "build": "write", "save": "write",
    "edit": "update", "modify": "update", "fix": "update", "refactor": "update", "patch": "update",
    "secure": "update", "harden": "update", "upgrade": "update", "replace": "update", "enhance": "update",
    "find": "search", "grep": "search", "lookup": "search", "locate": "search", "web_search": "search", "query": "search",
    "remove": "delete", "rm": "delete", "unlink": "delete",
}
_VALID_TASK_TYPES = {"read", "write", "search", "update", "delete", "other"}
_PRIORITY_ALIASES = {"urgent": "critical", "blocker": "critical", "p0": "critical", "important": "high", "p1": "high", "normal": "medium", "moderate": "medium", "default": "medium", "p2": "medium", "minor": "low", "p3": "low"}
_STATUS_ALIASES = {"todo": "pending", "queued": "pending", "new": "pending", "running": "in_progress", "active": "in_progress", "retry": "in_progress", "incomplete": "in_progress", "complete": "done", "completed": "done", "success": "done", "finished": "done", "error": "failed", "fail": "failed", "skip": "skipped", "canceled": "cancelled"}


def _coerce_str_list(val: Any) -> List[str]:
    if val is None:
        return []
    if isinstance(val, (list, tuple, set)):
        return [str(x).strip() for x in val if x is not None and str(x).strip()]
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return []
        if "," in s and not s.startswith("{"):
            return [p.strip() for p in s.split(",") if p.strip()]
        return [s]
    return [str(val).strip()] if str(val).strip() else []


def _normalize_enum(val: Any, valid: set, aliases: Dict[str, str], default: str) -> str:
    if val is None:
        return default
    raw = str(getattr(val, "value", val)).strip().lower()
    if raw in valid:
        return raw
    return aliases.get(raw, default)


def _clamp_rating(val: Any, default: int = 8) -> int:
    try:
        return max(1, min(10, int(float(val))))
    except Exception:
        return default


class ToolCall(BaseModel):
    tool_name: str = Field(..., description="Name of the tool to execute")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="Arguments to pass to the tool")

    if model_validator is not None:
        @model_validator(mode="before")
        @classmethod
        def _coerce_tool_call(cls, data: Any) -> Any:
            if not isinstance(data, dict):
                return data
            d = dict(data)
            if "tool_name" not in d:
                for alt in ("name", "tool", "function"):
                    if alt in d and d[alt]:
                        d["tool_name"] = d.pop(alt)
                        break
            if "arguments" not in d:
                for alt in ("args", "kwargs", "parameters", "params", "input"):
                    if alt in d:
                        d["arguments"] = d.pop(alt)
                        break
            raw_args = d.get("arguments")
            if raw_args is None:
                d["arguments"] = {}
            elif isinstance(raw_args, str):
                try:
                    parsed = json.loads(raw_args, strict=False)
                    d["arguments"] = parsed if isinstance(parsed, dict) else {}
                except Exception:
                    d["arguments"] = {}
            return d

ToolCallSchema = ToolCall


class TaskExecutionSchema(BaseModel):
    thought_process: str = Field(default="", description="Detailed analysis of the problem and the step-by-step approach to solve it")
    expected_outcome: Optional[str] = Field(default=None, description="Expected observable outcome of running tools_to_call before execution")
    tools_to_call: List[ToolCall] = Field(default_factory=list, description="Specific tool calls required to enact the solution")
    direct_response: Optional[str] = Field(default=None, description="Direct answer or finding when no further tool calls are needed")
    is_complete: bool = Field(default=False, description="Whether the task objective is satisfied by this step")

    if model_validator is not None:
        @model_validator(mode="before")
        @classmethod
        def _coerce_execution(cls, data: Any) -> Any:
            if not isinstance(data, dict):
                return data
            d = dict(data)
            if "tools_to_call" not in d:
                for alt in ("tool_calls", "actions", "calls"):
                    if alt in d:
                        d["tools_to_call"] = d.pop(alt)
                        break
            tc = d.get("tools_to_call")
            if tc is None:
                d["tools_to_call"] = []
            elif isinstance(tc, dict):
                d["tools_to_call"] = [tc]
            if d.get("direct_response") is None:
                for alt in ("response", "answer", "final_answer"):
                    if isinstance(d.get(alt), str) and d[alt].strip():
                        d["direct_response"] = d[alt]
                        break
            return d


class ActionWatchSchema(BaseModel):
    matches_expectation: bool = Field(default=True, description="Whether actual tool outputs and disk state satisfy the expected result")
    actual_summary: str = Field(default="", description="Factual summary of what actually occurred on disk/stdout")
    discrepancies: List[str] = Field(default_factory=list, description="Specific gaps or errors between expected and actual outcome")
    corrective_action: Optional[str] = Field(default=None, description="Concrete fix to apply on retry if mismatches exist")
    rating: int = Field(default=8, ge=1, le=10, description="Quality rating from 1 to 10")

    if model_validator is not None:
        @model_validator(mode="before")
        @classmethod
        def _coerce_watch(cls, data: Any) -> Any:
            if not isinstance(data, dict):
                return data
            d = dict(data)
            if "discrepancies" in d:
                d["discrepancies"] = _coerce_str_list(d.get("discrepancies"))
            if "rating" in d:
                d["rating"] = _clamp_rating(d.get("rating"), default=8)
            return d


class ActionSchema(BaseModel):
    action_id: str = Field(default_factory=generate_unique_id, description="Unique ID for the action")
    task_id: str = Field(default="", description="Associated task ID")
    thought_process: str = Field(default="", description="Reasoning behind the action")
    action_plan: Optional[str] = Field(default=None, description="Summary of the action plan")
    tools_to_call: List[ToolCall] = Field(default_factory=list, description="Tools to execute")


class ReflectionSchema(BaseModel):
    status: TaskStatus = Field(default=TaskStatus.DONE, description="The evaluation status: 'done', 'failed', or 'in_progress'")
    feedback: str = Field(default="", description="Detailed feedback on the runtime output and what needs to happen next")
    rating: int = Field(default=8, ge=1, le=10, description="Rating of the execution from 1 to 10")
    is_task_complete: bool = Field(default=True, description="Whether the entire task is now fully complete")
    summary: Optional[str] = Field(default=None, description="Concise summary of what this task accomplished")
    needs_replan: bool = Field(default=False, description="Whether remaining tasks in the plan need adjustment")
    confidence: float = Field(default=0.8, ge=0.0, le=1.0, description="Confidence score of evaluation")
    reasoning: str = Field(default="", description="Detailed reasoning for the evaluation")

    if model_validator is not None:
        @model_validator(mode="before")
        @classmethod
        def _coerce_reflection(cls, data: Any) -> Any:
            if not isinstance(data, dict):
                return data
            d = dict(data)
            if "status" in d:
                d["status"] = _normalize_enum(d.get("status"), {e.value for e in TaskStatus}, _STATUS_ALIASES, "done")
            if "rating" in d:
                d["rating"] = _clamp_rating(d.get("rating"), default=8)
            if "confidence" in d:
                try:
                    d["confidence"] = max(0.0, min(1.0, float(d["confidence"])))
                except Exception:
                    d["confidence"] = 0.8
            return d


class TaskSchema(BaseModel):
    task_id: str = Field(default_factory=generate_unique_id, description="Unique ID for the task")
    task_type: TaskType = Field(default=TaskType.OTHER, description="The operational archetype of the task")
    priority: TaskPriority = Field(default=TaskPriority.MEDIUM, description="Execution priority")
    description: str = Field(..., description="What the task requires")
    expected_result: Optional[str] = Field(default=None, description="Explicit expected outcome or verifiable post-state before acting")
    target_files: List[str] = Field(default_factory=list, description="Workspace files expected to be read, created, or modified")
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="Task workflow status")
    dependencies: List[str] = Field(default_factory=list, description="List of task_ids that must complete before this task")

    if model_validator is not None:
        @model_validator(mode="before")
        @classmethod
        def _coerce_task(cls, data: Any) -> Any:
            if not isinstance(data, dict):
                return data
            d = dict(data)
            if "task_id" in d and d["task_id"] is not None:
                d["task_id"] = str(d["task_id"])
            if "task_type" in d:
                d["task_type"] = _normalize_enum(d.get("task_type"), _VALID_TASK_TYPES, _TASK_TYPE_ALIASES, "other")
            if "priority" in d:
                d["priority"] = _normalize_enum(d.get("priority"), {e.value for e in TaskPriority}, _PRIORITY_ALIASES, "medium")
            if "status" in d:
                d["status"] = _normalize_enum(d.get("status"), {e.value for e in TaskStatus}, _STATUS_ALIASES, "pending")
            if "target_files" in d:
                d["target_files"] = _coerce_str_list(d.get("target_files"))
            if "dependencies" in d:
                d["dependencies"] = _coerce_str_list(d.get("dependencies"))
            if not d.get("description"):
                for alt in ("task", "title", "name", "action", "goal"):
                    if isinstance(d.get(alt), str) and d[alt].strip():
                        d["description"] = d[alt].strip()
                        break
            return d


class PlanSchema(BaseModel):
    objective: str = Field(..., description="The main overall objective")
    tasks: List[TaskSchema] = Field(default_factory=list, description="List of ordered tasks to complete the objective")

    if model_validator is not None:
        @model_validator(mode="before")
        @classmethod
        def _coerce_plan(cls, data: Any) -> Any:
            if not isinstance(data, dict):
                return data
            d = dict(data)
            if not d.get("objective"):
                for alt in ("goal", "plan_objective", "summary"):
                    if isinstance(d.get(alt), str) and d[alt].strip():
                        d["objective"] = d[alt].strip()
                        break
                else:
                    d["objective"] = "Execute user request"
            raw_tasks = d.get("tasks")
            if raw_tasks is None:
                d["tasks"] = []
            elif isinstance(raw_tasks, dict):
                d["tasks"] = [raw_tasks]
            return d

ProblemSchema = Problem
SolutionSchema = Solution

# =========================================================================
# ACTOR SCHEMAS
# =========================================================================

class ActorInputSchema(BaseModel):
    """Strict input schema for the Actor module to execute a task."""
    task_id: str = Field(default_factory=generate_unique_id, description="Unique ID of the task being executed")
    task_type: TaskType = Field(default=TaskType.OTHER, description="The type of the task")
    tool_name: str = Field(..., description="Name of the tool to execute")
    payload: Dict[str, Any] = Field(default_factory=dict, description="Arguments for the tool")

class ActorOutputSchema(BaseModel):
    """Strict output schema for the Actor module returning a task execution."""
    task_id: str = Field(..., description="Unique ID of the task that was executed")
    tool_name: str = Field(default="unknown", description="Name of the tool that was executed")
    success: bool = Field(..., description="Whether the actor successfully completed the task")
    result: Optional[Dict[str, Any]] = Field(None, description="The normalized tool output payload")
    error_context: Optional[str] = Field(None, description="Additional actor-level error context")
    expected_outcome: Optional[str] = Field(default=None, description="Expected outcome declared before tool execution")
    artifacts: Dict[str, Any] = Field(default_factory=dict, description="Tracked files read/modified and tool counts")
    reflection: Optional[Any] = Field(None, description="Optional reflection output from the Reflector module")

class ActorToReflectorSchema(BaseModel):
    """Schema to send the Actor's tool execution output directly to the Reflector."""
    task_context: TaskSchema = Field(..., description="The full task context")
    actor_output: ActorOutputSchema = Field(..., description="The execution result of the actor")

# Base Tool Schemas
class BaseTaskInputSchema(BaseModel):
    task_id: Optional[str] = Field(default=None, description="Optional task ID")
    task_description: Optional[str] = Field(default=None, description="Optional task description")
    task_type: Optional[TaskType] = Field(default=None, description="Optional task type")

class BaseTaskOutputSchema(BaseModel):
    task_id: Optional[str] = Field(default=None, description="Optional task ID")
    success: bool = Field(default=True, description="Whether the task execution was successful")
    error: Optional[str] = Field(default=None, description="Error message if the task failed")
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
