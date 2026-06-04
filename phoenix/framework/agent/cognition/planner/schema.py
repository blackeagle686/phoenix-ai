from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Union
from enum import Enum
from uuid import UUID

class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    SKIPPED = "skipped"

from enum import Enum

class TaskType(str, Enum):
    # =========================================================================
    # 1. STANDARD CRUD & FILE SYSTEM I/O (OS / Storage)
    # =========================================================================
    READ = "read"              # Generic file/stream reading (disk, RAM)
    WRITE = "write"            # Generic file/stream writing (disk, RAM)
    SEARCH = "search"          # File system directory traversal / text regex search
    UPDATE = "update"          # Modifying existing blocks, records, or file segments
    DELETE = "delete"          # Removing file descriptors, blocks, or database entries
    BLOCK_READ = "block_read"  # Low-level direct disk sector read (NVMe/SSD)
    BLOCK_WRITE = "block_write" # Low-level direct disk sector write (NVMe/SSD)
    MMAP_IO = "mmap_io"        # Mapping file descriptors directly into virtual memory
    
    # =========================================================================
    # 2. NETWORK & COMMUNICATIONS I/O (OS / Cloud / IPC)
    # =========================================================================
    NET_SEND = "net_send"      # Outbound network packet transfer (TCP, UDP, WebSockets)
    NET_RECV = "net_recv"      # Inbound network packet capture
    IPC_PIPE = "ipc_pipe"      # Moving data between processes (Pipes, Unix Domain Sockets)
    IPC_SHARE = "ipc_share"    # Allocating/accessing Inter-Process Shared Memory
    RPC_CALL = "rpc_call"      # Remote Procedure Call invocation (gRPC, REST API)

    # =========================================================================
    # 3. AI, EMBEDDINGS & MACHINE LEARNING I/O
    # =========================================================================
    BATCH_LOAD = "batch_load"       # High-throughput data ingestion from storage to RAM
    TENSOR_STREAM = "tensor_stream" # Streaming multi-dimensional arrays through NN layers
    VECTOR_SEARCH = "vector_search" # K-Nearest Neighbor (KNN/ANN) queries on vector DBs
    VRAM_SHUTTLE = "vram_shuttle"   # Moving data across PCIe lanes between system RAM & VRAM
    TOKEN_STREAM = "token_stream"   # Sequential token-by-token real-time LLM text generation

    # =========================================================================
    # 4. HARDWARE-LEVEL & LOW-LEVEL BUS I/O (Embedded / OS Kernels)
    # =========================================================================
    DMA_TRANSFER = "dma_transfer"   # Offloading block moves to a Direct Memory Access controller
    INTERRUPT_REQ = "interrupt_req" # Handling hardware signal events via ISRs
    PORT_IN = "port_in"             # Reading from dedicated isolated hardware I/O ports
    PORT_OUT = "port_out"           # Writing to dedicated isolated hardware I/O ports
    MEM_MAPPED_IN = "mem_mapped_in" # Reading hardware states from mapped physical RAM space

    # =========================================================================
    # 5. MISSION-CRITICAL & VEHICLE SYSTEMS I/O (Planes, Ships, Aircraft)
    # =========================================================================
    BUS_BROADCAST = "bus_broadcast" # Sending telemetry over avionics/maritime buses (ARINC 429, CAN)
    BUS_LISTEN = "bus_listen"       # Continuous interception of bus datagrams
    ADC_SAMPLE = "adc_sample"       # Sampling physical realities via Analog-to-Digital converters
    DAC_ACTUATE = "dac_actuate"     # Sending voltage commands via Digital-to-Analog converters
    PWM_OUTPUT = "pwm_output"       # Generating high-frequency square waves for motor controls
    SENSOR_POLL = "sensor_poll"     # Aggregating data across high-frequency IMUs, Radar, and Sonar
    WATCHDOG_PING = "watchdog_ping" # Periodic heartbeat signal to clear safety timer circuits

class TaskPriority(str, Enum):
    CRITICAL = "critical"  # Real-time systems / Watchdogs
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class FileOperation(str, Enum):
    READ = "read"
    WRITE = "write"
    APPEND = "append"
    SEARCH = "search"
    REPLACE = "replace"
    DELETE = "delete"
    CHMOD = "chmod"       # Crucial for OS file permissions

class FileIOParams(BaseModel):
    file_path: str = Field(..., description="Absolute path to the target file")
    operation: FileOperation = Field(..., description="The specific file sub-operation")
    
    # Text and binary data handling
    content: Optional[Union[str, bytes]] = Field(None, description="Raw text or binary bytes to write")
    encoding: str = Field(default="utf-8", description="File text encoding style (e.g., utf-8, ascii, latin1)")
    
    # Fine-grained file system positioning
    offset: Optional[int] = Field(None, description="Byte position to seek to before reading or writing")
    length: Optional[int] = Field(None, description="Number of bytes to read if performing a partial block read")
    
    # Query operations
    search_query: Optional[str] = Field(None, description="Regex or text token to look up inside the file")
    replace_query: Optional[str] = Field(None, description="The content used to substitute the search match")
    
    # OS Level Flags
    permissions: Optional[str] = Field(None, description="Linux octal permission string (e.g., '0o755' or '0o644')")

class Prompt(BaseModel):
    id: UUID = Field(default_factory=uuid.uuid4, description="Unique identifier for the user prompt")
    user_id: UUID = Field(default_factory=uuid.uuid4, description="Unique identifier for the user")
    project_id: UUID = Field(default_factory=uuid.uuid4, description="Unique identifier for the project")
    user_message: str = Field(..., description="User message")
    system_message: str = Field(..., description="System message")
    tokens_length: int = Field(..., description="Tokens length")

class Task(BaseModel):
    # --- Identifiers & Relationships ---
    prompt_id: UUID = Field(default_factory=uuid4, description="Unique identifier for the parent session/prompt request")
    task_id: str = Field(..., description="Unique deterministic identifier for this specific task")
    dependencies: List[str] = Field(default_factory=list, description="List of task_ids that must complete successfully first")
    
    # --- Strongly Typed Meta Elements ---
    task_type: TaskType = Field(..., description="The specific systemic I/O operation archetype")
    priority: TaskPriority = Field(default=TaskPriority.MEDIUM, description="The execution urgency tier")
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="Current workflow state machine status")
    
    # --- Descriptive Data (For Humans & LLM Reasoning) ---
    task_title: str = Field(..., description="Short title of the task")
    description: str = Field(..., description="Verbose description detailing task goals")
    task_summary: Optional[str] = Field(None, description="Post-execution summary populated after completion")
    
    # --- Execution Data (The Machine-Readable Payload) ---
    payload: Dict[str, Any] = Field(default_factory=dict, description="Input parameters required by the driver (e.g., {'ip': '127.0.0.1', 'bytes': b'...'})")
    result: Optional[Dict[str, Any]] = Field(None, description="Output returned by the executing module/driver")
    error: Optional[str] = Field(None, description="Error tracking message if status shifts to FAILED")

    # --- Routing & Governance ---
    created_by: str = Field(..., description="The identifier of the agent/orchestrator that generated this task")
    assigned_to: Optional[str] = Field(None, description="The targeted agent, execution worker pool, or hardware driver")
    
    # --- Time Guarantees (SLA / Deadlines) ---
    timeout: float = Field(default=30.0, description="Max execution time window in seconds before the task is forcefully killed")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Timestamp when the task was initially queued")
    executed_at: Optional[datetime] = Field(None, description="Timestamp when the execution worker actually started processing")

    class Config:
        use_enum_values = True  # Allows smooth JSON serialization when saving to DBs or sending over APIs




class FileContent(BaseModel):
    file_path: str = Field(..., description="Path to the file")
    content_block: str = Field(..., description="Raw text or code content of this specific block")
    
    # تحديد المواقع (Structural Indexing)
    from_line: int = Field(..., description="1-indexed line number where this block starts")
    to_line: int = Field(..., description="1-indexed line number where this block ends (inclusive)")
    
    # حرج جداً لأنظمة الـ AI والـ RAG
    token_count: Optional[int] = Field(None, description="Calculated LLM token count for this specific block")
    overlap_lines: int = Field(default=0, description="Number of lines duplicated from the previous block for context preservation")
    
    # المعرفة الدلالية (Semantic Metadata)
    block_summary: Optional[str] = Field(None, description="AI-generated summary of this content block")
    vector_id: Optional[str] = Field(None, description="Reference ID if this block is embedded in a Vector Database")
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





