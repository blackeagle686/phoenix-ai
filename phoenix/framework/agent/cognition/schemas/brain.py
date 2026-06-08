
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any

from phoenix.framework.agent.cognition.planner.schema import (
    TaskType,
    TaskPriority,
    TaskStatus,
    ProblemComplexity,
    SolutionType,
    FileOperation
)

class ProblemDefinition(BaseModel):
    problem_id: str = Field(..., description="Unique ID for this problem") # system defined
    description: str = Field(..., description="Clear description of the problem to solve") # llm defined 
    related_context: str = Field(..., description="Context or findings related to this problem") # llm defined
    files_to_analyze: List[str] = Field(default_factory=list, description="List of files to look into for this problem") # llm defined
    complexity: ProblemComplexity = Field(default=ProblemComplexity.MEDIUM, description="The assessed complexity of this problem") # llm defined

class ProblemSchema(BaseModel):
    task_id: str = Field(..., description="The ID of the task these problems belong to")
    problems: List[ProblemDefinition] = Field(default_factory=list, description="List of defined problems to solve the task")

class SolutionDefinition(BaseModel):
    solution_id: str = Field(..., description="Unique ID for this solution")
    problem_id: str = Field(..., description="The ID of the problem this solves")
    solution_type: SolutionType = Field(default=SolutionType.CODE, description="The archetype of the solution")
    approach: str = Field(..., description="Detailed algorithmic or structural approach to solving the problem")
    required_tools: List[str] = Field(default_factory=list, description="Names of the tools required to enact this solution")

class SolutionSchema(BaseModel):
    task_id: str = Field(..., description="The ID of the task")
    solutions: List[SolutionDefinition] = Field(default_factory=list, description="List of solutions corresponding to problems")

class IOOperation(BaseModel):
    operation: FileOperation = Field(..., description="The type of operation (e.g., 'create', 'create_dir', 'edit', 'delete', 'read')")
    file_path: str = Field(..., description="Absolute or relative path to the file")
    content: Optional[str] = Field(None, description="The content to write or edit if applicable")

class ToolCall(BaseModel):
    tool_name: str = Field(..., description="Name of the tool to execute")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="Arguments to pass to the tool")

class ActionSchema(BaseModel):
    solution_id: str = Field(..., description="The ID of the solution being enacted")
    tools_to_call: List[ToolCall] = Field(default_factory=list, description="Specific tool calls with arguments")
    io_operations: List[IOOperation] = Field(default_factory=list, description="List of explicit file I/O operations")
    action_plan: str = Field(..., description="A short summary of what this action will do")

class ReflectionSchema(BaseModel):
    status: TaskStatus = Field(..., description="The evaluation status: 'done', 'failed', or 'in_progress'")
    feedback: str = Field(..., description="Detailed feedback on the runtime output and what needs to happen next")
    rating: int = Field(..., ge=1, le=10, description="Rating of the execution from 1 to 10")
    is_task_complete: bool = Field(..., description="Whether the entire task is now fully complete")

class TaskSchema(BaseModel):
    task_id: str = Field(..., description="Unique ID for the task")
    task_type: TaskType = Field(default=TaskType.OTHER, description="The operational archetype of the task")
    priority: TaskPriority = Field(default=TaskPriority.MEDIUM, description="Execution priority")
    description: str = Field(..., description="What the task requires")
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="Task workflow status")

class PlanSchema(BaseModel):
    objective: str = Field(..., description="The main overall objective")
    tasks: List[TaskSchema] = Field(default_factory=list, description="List of ordered tasks to complete the objective")
