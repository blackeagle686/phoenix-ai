import re

with open('phoenix/framework/agent/cognition/planner/schema.py', 'r') as f:
    content = f.read()

# Fix FileIOMeta inheritance
content = content.replace("class FileIOMeta(BaseFileMeta):", "class FileIOMeta(BaseModel):")

# Add BaseFileTaskOutputSchema
base_out_schema = """
class BaseFileTaskOutputSchema(BaseTaskOutputSchema):
    file_path: str = Field(..., description="Path to the associated file")
"""
content = content.replace("class BaseFileTaskInputSchema(BaseTaskInputSchema):\n    file_meta: BaseFileMeta = Field(..., description=\"Meta information of the file\")", "class BaseFileTaskInputSchema(BaseTaskInputSchema):\n    file_meta: BaseFileMeta = Field(..., description=\"Meta information of the file\")\n" + base_out_schema)

# 1. Remove duplicate `file_path` from tasks inheriting from BaseFileTaskInputSchema
tasks_to_fix = [
    "FileReadTask", "FileWriteTask", "FileSearchTask", "FileUpdateTask", 
    "PythonAnalysisTask", "MultiBlockUpdateTask", "CodeCompileTask"
]
for task in tasks_to_fix:
    # regex to remove file_path line inside the class
    pattern = rf"(class {task}\(BaseFileTaskInputSchema\):[\s\S]*?)(\s+file_path:\ str = Field\([^\n]+\n)"
    content = re.sub(pattern, r"\1", content)

# 2. Change results to BaseFileTaskOutputSchema and remove `file_path`
results_to_fix = [
    "FileReadResult", "FileWriteResult", "FileSearchResult", "FileUpdateResult",
    "PythonAnalysisResult", "MultiBlockUpdateResult", "CodeCompileResult"
]
for res in results_to_fix:
    content = content.replace(f"class {res}(BaseTaskOutputSchema):", f"class {res}(BaseFileTaskOutputSchema):")
    pattern = rf"(class {res}\(BaseFileTaskOutputSchema\):[\s\S]*?)(\s+file_path:\ str = Field\([^\n]+\n)"
    content = re.sub(pattern, r"\1", content)

# 3. Remove duplicate `success` and `error` from any BaseTaskOutputSchema or BaseFileTaskOutputSchema
content = re.sub(r'(\s+success:\ bool = Field\([^\n]+\n)', '', content)
content = re.sub(r'(\s+error:\ Optional\[str\] = Field\([^\n]+\n)', '', content)

# But wait! We need to make sure we don't remove `success` and `error` from the base classes themselves!
# Let's restore them in BaseTaskOutputSchema and Task
base_task_out = """class BaseTaskOutputSchema(BaseModel):
    task_id: str = Field(..., description="Task ID")
    success: bool = Field(..., description="Whether the task execution was successful")
    error: Optional[str] = Field(None, description="Error message if the task failed")"""

content = re.sub(r'class BaseTaskOutputSchema\(BaseModel\):[\s\S]*?(?=class BaseFileTaskInputSchema)', base_task_out + "\n\n", content)

# We also need to restore `error` in Task
task_replacement = """    result: Optional[Dict[str, Any]] = Field(None, description="Output returned by the executing module/driver") # result of the task
    error: Optional[str] = Field(None, description="Error tracking message if status shifts to FAILED")"""
content = re.sub(r'    result: Optional\[Dict\[str, Any\]\] = Field.*?# result of the task\n', task_replacement + '\n', content)

# If a class becomes empty, add `pass`
content = re.sub(r'(class \w+\(\w+\):)\n(?=\n|#|class)', r'\1\n    pass\n', content)

with open('phoenix/framework/agent/cognition/planner/schema_clean.py', 'w') as f:
    f.write(content)
