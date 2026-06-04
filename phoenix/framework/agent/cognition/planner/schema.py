from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Union, Any
from enum import Enum
from uuid import UUID, uuid4
from datetime import datetime
from phoenix.framework.agent.cognition.reflector.schema import BaseReflectorMeta

class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    SKIPPED = "skipped"

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
    OTHER = "other"                 # Generic fallback for cognitive actions that do not fit into OS/System categories
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
    id: UUID = Field(default_factory=uuid4, description="Unique identifier for the user prompt")
    user_id: Optional[UUID] = Field(None, description="Unique identifier for the user")
    project_id: Optional[UUID] = Field(None, description="Unique identifier for the project")
    user_message: str = Field(..., description="User message")
    system_message: str = Field(..., description="System message")
    tokens_length: int = Field(..., description="Tokens length")

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
    task_id: str = Field(..., description="Unique deterministic identifier for this specific task")
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

class FileContent(BaseModel):
    file_path: str = Field(..., description="Path to the file")
    content_block: str = Field(..., description="Raw text or code content of this specific block")
    from_line: int = Field(..., description="1-indexed line number where this block starts")
    to_line: int = Field(..., description="1-indexed line number where this block ends (inclusive)")
    token_count: Optional[int] = Field(None, description="Calculated LLM token count for this specific block")
    overlap_lines: int = Field(default=0, description="Number of lines duplicated from the previous block for context preservation")
    block_summary: Optional[str] = Field(None, description="AI-generated summary of this content block")
    vector_id: Optional[str] = Field(None, description="Reference ID if this block is embedded in a Vector Database")

class File(BaseModel):
    file_id: str = Field(..., description="Unique deterministic UUID or hash of the file path")
    file_path: str = Field(..., description="Absolute or relative path to the file")
    file_type: str = Field(..., description="File extension or MIME type (e.g., 'py', 'json', 'txt')")
    content: List[FileContent] = Field(default_factory=list, description="Ordered list of segmented content blocks")
    total_lines: int = Field(..., description="Total line count of the source file")
    total_blocks: int = Field(..., description="Total number of split blocks")
    total_tokens: Optional[int] = Field(None, description="Aggregate token count of the entire file")
    file_hash: Optional[str] = Field(None, description="MD5/SHA256 checksum to detect external modifications")
    encoding: str = Field(default="utf-8", description="File text encoding (e.g., utf-8, ascii, utf-16)")
    file_summary: Optional[str] = Field(None, description="High-level systemic summary of the overall file purpose")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Custom extensions like OS permissions, owner, or git branch")

    class Config:
        populate_by_name = True

class BaseFileMeta(BaseModel):
    file_path: str = Field(..., description="Path of the file")
    file_name: str = Field(..., description="Name of the file")
    lines_count: int = Field(..., description="Number of lines in the file")
    file_size: int = Field(..., description="Size of the file in bytes")
    file_type: str = Field(..., description="Type of the file")
    last_modified_time: int = Field(..., description="Last modified time of the file in seconds since epoch")

class FileUpdateStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"

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

class FileIOMeta(BaseModel):
    file_meta: BaseFileMeta = Field(..., description="Configuration of the file")
    from_line: int = Field(..., description="Start line number of the file")
    to_line: int = Field(..., description="End line number of the file")
    status: FileUpdateStatus = Field(..., description="Status of the file update")
