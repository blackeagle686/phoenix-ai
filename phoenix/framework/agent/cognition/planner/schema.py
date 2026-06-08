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

# =========================================================================
# TASK TYPE SPECIFIC SCHEMAS
# =========================================================================

# 1. STANDARD CRUD & FILE SYSTEM I/O
class ReadTaskSchema(BaseModel):
    file_path: str = Field(..., description="Target file path to read")
    offset: Optional[int] = Field(None, description="Byte offset to start reading from")
    length: Optional[int] = Field(None, description="Number of bytes to read")

class WriteTaskSchema(BaseModel):
    file_path: str = Field(..., description="Target file path to write to")
    content: Union[str, bytes] = Field(..., description="Content to write")
    append: bool = Field(False, description="Whether to append to the file instead of overwriting")

class SearchTaskSchema(BaseModel):
    directory_path: str = Field(..., description="Directory to search in")
    query: str = Field(..., description="Regex or string query to search for")
    file_pattern: Optional[str] = Field(None, description="File glob pattern to restrict search")

class UpdateTaskSchema(BaseModel):
    file_path: str = Field(..., description="Target file path to update")
    search_query: str = Field(..., description="Text or block to find")
    replace_content: str = Field(..., description="Content to replace with")

class DeleteTaskSchema(BaseModel):
    target_path: str = Field(..., description="File or directory path to delete")
    recursive: bool = Field(False, description="Whether to recursively delete directories")

class BlockReadTaskSchema(BaseModel):
    device_path: str = Field(..., description="Block device path (e.g., /dev/nvme0n1)")
    sector_offset: int = Field(..., description="Starting sector")
    num_sectors: int = Field(..., description="Number of sectors to read")

class BlockWriteTaskSchema(BaseModel):
    device_path: str = Field(..., description="Block device path")
    sector_offset: int = Field(..., description="Starting sector")
    data: bytes = Field(..., description="Raw bytes to write to sectors")

class MmapIoTaskSchema(BaseModel):
    file_path: str = Field(..., description="File to map into memory")
    size: int = Field(..., description="Size of mapping")
    offset: int = Field(0, description="Offset in file")

# 2. NETWORK & COMMUNICATIONS I/O
class NetSendTaskSchema(BaseModel):
    destination: str = Field(..., description="IP or hostname to send to")
    port: int = Field(..., description="Target port")
    protocol: str = Field("TCP", description="Protocol to use (TCP/UDP)")
    data: bytes = Field(..., description="Data to send")

class NetRecvTaskSchema(BaseModel):
    port: int = Field(..., description="Port to listen on")
    protocol: str = Field("TCP", description="Protocol to use (TCP/UDP)")
    timeout: float = Field(..., description="Timeout in seconds")

class IpcPipeTaskSchema(BaseModel):
    pipe_name: str = Field(..., description="Name or path of the named pipe")
    data: Optional[bytes] = Field(None, description="Data to write (if writing)")
    read_length: Optional[int] = Field(None, description="Bytes to read (if reading)")

class IpcShareTaskSchema(BaseModel):
    shm_name: str = Field(..., description="Shared memory segment name")
    size: int = Field(..., description="Size of shared memory block")
    write_data: Optional[bytes] = Field(None, description="Data to write to shared memory")

class RpcCallTaskSchema(BaseModel):
    endpoint: str = Field(..., description="RPC or REST endpoint URL")
    method: str = Field(..., description="Method name to invoke")
    params: Dict[str, Any] = Field(default_factory=dict, description="Parameters for the RPC call")

# 3. AI, EMBEDDINGS & MACHINE LEARNING I/O
class BatchLoadTaskSchema(BaseModel):
    source_uri: str = Field(..., description="URI of the data source")
    batch_size: int = Field(..., description="Number of items per batch")
    destination_buffer: str = Field(..., description="Target buffer or memory segment")

class TensorStreamTaskSchema(BaseModel):
    model_id: str = Field(..., description="Identifier for the neural network model")
    input_tensor: Any = Field(..., description="Input tensor data")
    expected_shape: List[int] = Field(..., description="Expected shape of the output tensor")

class VectorSearchTaskSchema(BaseModel):
    collection_name: str = Field(..., description="Name of the vector collection")
    query_vector: List[float] = Field(..., description="Vector to search for")
    top_k: int = Field(10, description="Number of nearest neighbors to return")

class VramShuttleTaskSchema(BaseModel):
    direction: str = Field(..., description="'host_to_device' or 'device_to_host'")
    size_bytes: int = Field(..., description="Amount of data to transfer")
    memory_pointer: str = Field(..., description="Reference to the memory address")

class TokenStreamTaskSchema(BaseModel):
    prompt: str = Field(..., description="Text prompt for the LLM")
    max_tokens: int = Field(..., description="Maximum tokens to generate")
    stream: bool = Field(True, description="Whether to stream the response")

# 4. HARDWARE-LEVEL & LOW-LEVEL BUS I/O
class DmaTransferTaskSchema(BaseModel):
    source_addr: str = Field(..., description="Physical source address")
    dest_addr: str = Field(..., description="Physical destination address")
    transfer_size: int = Field(..., description="Size in bytes to transfer")

class InterruptReqTaskSchema(BaseModel):
    irq_number: int = Field(..., description="IRQ line number")
    handler_id: str = Field(..., description="Identifier of the ISR to trigger or register")

class PortInTaskSchema(BaseModel):
    port_address: str = Field(..., description="I/O port address")
    read_size: int = Field(1, description="Number of bytes to read")

class PortOutTaskSchema(BaseModel):
    port_address: str = Field(..., description="I/O port address")
    data: bytes = Field(..., description="Data to write to the port")

class MemMappedInTaskSchema(BaseModel):
    base_address: str = Field(..., description="Base physical address")
    offset: int = Field(..., description="Offset from base address")
    read_size: int = Field(..., description="Number of bytes to read")

# 5. MISSION-CRITICAL & VEHICLE SYSTEMS I/O
class BusBroadcastTaskSchema(BaseModel):
    bus_type: str = Field(..., description="e.g., 'CAN', 'ARINC_429'")
    message_id: str = Field(..., description="Identifier for the broadcast message")
    payload: bytes = Field(..., description="Raw message payload")

class BusListenTaskSchema(BaseModel):
    bus_type: str = Field(..., description="e.g., 'CAN', 'ARINC_429'")
    filter_id: Optional[str] = Field(None, description="Optional ID to filter on")
    duration_sec: float = Field(..., description="Time to listen")

class AdcSampleTaskSchema(BaseModel):
    channel: int = Field(..., description="ADC channel number")
    sample_rate: int = Field(..., description="Samples per second")
    duration_sec: float = Field(..., description="Duration to sample")

class DacActuateTaskSchema(BaseModel):
    channel: int = Field(..., description="DAC channel number")
    voltage_level: float = Field(..., description="Target voltage to output")

class PwmOutputTaskSchema(BaseModel):
    pin: int = Field(..., description="PWM pin number")
    frequency_hz: float = Field(..., description="Frequency in Hertz")
    duty_cycle: float = Field(..., description="Duty cycle percentage (0.0 to 1.0)")

class SensorPollTaskSchema(BaseModel):
    sensor_id: str = Field(..., description="Identifier for the sensor (e.g., IMU_1)")
    poll_rate_hz: int = Field(..., description="Polling rate in Hertz")

class WatchdogPingTaskSchema(BaseModel):
    timer_id: str = Field(..., description="Identifier of the watchdog timer")
    reset_value: Optional[int] = Field(None, description="Value to reset the timer to")

class OtherTaskSchema(BaseModel):
    action_name: str = Field(..., description="Name of the custom action")
    custom_params: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary parameters")

TaskPayloadType = Union[
    ReadTaskSchema, WriteTaskSchema, SearchTaskSchema, UpdateTaskSchema, DeleteTaskSchema,
    BlockReadTaskSchema, BlockWriteTaskSchema, MmapIoTaskSchema,
    NetSendTaskSchema, NetRecvTaskSchema, IpcPipeTaskSchema, IpcShareTaskSchema, RpcCallTaskSchema,
    BatchLoadTaskSchema, TensorStreamTaskSchema, VectorSearchTaskSchema, VramShuttleTaskSchema, TokenStreamTaskSchema,
    DmaTransferTaskSchema, InterruptReqTaskSchema, PortInTaskSchema, PortOutTaskSchema, MemMappedInTaskSchema,
    BusBroadcastTaskSchema, BusListenTaskSchema, AdcSampleTaskSchema, DacActuateTaskSchema, PwmOutputTaskSchema,
    SensorPollTaskSchema, WatchdogPingTaskSchema, OtherTaskSchema,
    Dict[str, Any]
]
    
class TaskPriority(str, Enum):
    CRITICAL = "critical"  # Real-time systems / Watchdogs
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
    payload: TaskPayloadType = Field(default_factory=dict, description="Input parameters required by the driver")
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
