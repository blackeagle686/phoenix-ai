from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any

class ProblemDefinition(BaseModel):
    problem_id: str = Field(..., description="Unique ID for this problem")
    description: str = Field(..., description="Clear description of the problem to solve")
    related_context: str = Field(..., description="Context or findings related to this problem")
    files_to_analyze: List[str] = Field(default_factory=list, description="List of files to look into for this problem")

class ProblemSchema(BaseModel):
    task_id: str = Field(..., description="The ID of the task these problems belong to")
    problems: List[ProblemDefinition] = Field(default_factory=list, description="List of defined problems to solve the task")

class SolutionDefinition(BaseModel):
    solution_id: str = Field(..., description="Unique ID for this solution")
    problem_id: str = Field(..., description="The ID of the problem this solves")
    approach: str = Field(..., description="Detailed algorithmic or structural approach to solving the problem")
    required_tools: List[str] = Field(default_factory=list, description="Names of the tools required to enact this solution")

class SolutionSchema(BaseModel):
    task_id: str = Field(..., description="The ID of the task")
    solutions: List[SolutionDefinition] = Field(default_factory=list, description="List of solutions corresponding to problems")

class IOOperation(BaseModel):
    operation: str = Field(..., description="The type of operation (e.g., 'create', 'edit', 'delete', 'read')")
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
    status: str = Field(..., description="The evaluation status: 'completed', 'failed', or 'in_progress'")
    feedback: str = Field(..., description="Detailed feedback on the runtime output and what needs to happen next")
    rating: int = Field(..., ge=1, le=10, description="Rating of the execution from 1 to 10")
    is_task_complete: bool = Field(..., description="Whether the entire task is now fully complete")

class TaskSchema(BaseModel):
    task_id: str = Field(..., description="Unique ID for the task")
    description: str = Field(..., description="What the task requires")
    status: str = Field(default="pending", description="Task status")

class PlanSchema(BaseModel):
    objective: str = Field(..., description="The main overall objective")
    tasks: List[TaskSchema] = Field(default_factory=list, description="List of ordered tasks to complete the objective")
