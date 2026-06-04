import re

with open('phoenix/framework/agent/cognition/planner/schema.py', 'r') as f:
    content = f.read()

# We'll use regex and string replacement to update the schemas.
# First, remove BaseTaskInputSchema and BaseTaskMeta since we'll redefine them.
content = re.sub(r'class BaseTaskInputSchema\(BaseModel\):.*?file_meta: BaseFileMeta = Field\(\.\.\., description="Meta information of the file"\)', '', content, flags=re.DOTALL)
content = re.sub(r'class BaseTaskMeta\(BaseModel\):.*?(?=\nclass WriteTask)', '', content, flags=re.DOTALL)

# Add our new base schemas after FileIOMeta
new_base_schemas = """
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
"""

content = content.replace("class FileIOMeta(BaseFileMeta):", new_base_schemas + "\nclass FileIOMeta(BaseFileMeta):")

# Now update individual tasks
content = content.replace("class FileReadTask(BaseModel):", "class FileReadTask(BaseFileTaskInputSchema):")
content = content.replace("class FileReadResult(BaseModel):", "class FileReadResult(BaseTaskOutputSchema):")

content = content.replace("class FileWriteTask(BaseModel):", "class FileWriteTask(BaseFileTaskInputSchema):")
content = content.replace("class FileWriteResult(BaseModel):", "class FileWriteResult(BaseTaskOutputSchema):")

content = content.replace("class FileSearchTask(BaseModel):", "class FileSearchTask(BaseFileTaskInputSchema):")
content = content.replace("class FileSearchResult(BaseModel):", "class FileSearchResult(BaseTaskOutputSchema):")

content = content.replace("class FileUpdateTask(BaseModel):", "class FileUpdateTask(BaseFileTaskInputSchema):")
content = content.replace("class FileUpdateResult(BaseModel):", "class FileUpdateResult(BaseTaskOutputSchema):")

content = content.replace("class WebSearchTask(BaseModel):", "class WebSearchTask(BaseTaskInputSchema):")
content = content.replace("class WebSearchResult(BaseModel):", "class WebSearchResult(BaseTaskOutputSchema):")

content = content.replace("class CodeExecutionTask(BaseModel):", "class CodeExecutionTask(BaseTaskInputSchema):")
content = content.replace("class CodeExecutionResult(BaseModel):", "class CodeExecutionResult(BaseTaskOutputSchema):")

content = content.replace("class PythonAnalysisTask(BaseModel):", "class PythonAnalysisTask(BaseFileTaskInputSchema):")
content = content.replace("class PythonAnalysisResult(BaseModel):", "class PythonAnalysisResult(BaseTaskOutputSchema):")

content = content.replace("class MultiBlockUpdateTask(BaseModel):", "class MultiBlockUpdateTask(BaseFileTaskInputSchema):")
content = content.replace("class MultiBlockUpdateResult(BaseModel):", "class MultiBlockUpdateResult(BaseTaskOutputSchema):")

content = content.replace("class CommandExecutionTask(BaseModel):", "class CommandExecutionTask(BaseTaskInputSchema):")
content = content.replace("class CommandExecutionResult(BaseModel):", "class CommandExecutionResult(BaseTaskOutputSchema):")

content = content.replace("class CodeCompileTask(BaseModel):", "class CodeCompileTask(BaseFileTaskInputSchema):")
content = content.replace("class CodeCompileResult(BaseModel):", "class CodeCompileResult(BaseTaskOutputSchema):")

with open('phoenix/framework/agent/cognition/planner/schema_new.py', 'w') as f:
    f.write(content)

