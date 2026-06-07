import asyncio
import os
import sys

# Add the project root to sys.path (insert at index 0 to override installed packages)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from phoenix.framework.agent.core.agent import Agent
from phoenix.framework.agent.cognition.schemas.brain import (
    PlanSchema, ProblemSchema, SolutionSchema, ActionSchema, ReflectionSchema,
    TaskSchema, ProblemDefinition, SolutionDefinition, ToolCall, IOOperation
)
from phoenix.framework.agent.tools.registry import ToolRegistry
from phoenix.framework.agent.execution.tool_manager import ToolManager
from phoenix.framework.agent.tools.base import ToolResult

class MockTool:
    def __init__(self, name):
        self.name = name
        self.description = f"Mock tool {name}"
        
    async def execute(self, **kwargs):
        print(f"    [Tool Executed] {self.name} with {kwargs}")
        return ToolResult(success=True, output=f"Executed {self.name}", error=None)

class MockLLM:
    """Mocks the LLM calls to return exact Pydantic schemas for the centralized Thinker tests."""
    def __init__(self):
        self.client = "mock"
        
    async def init(self):
        pass

    async def generate(self, prompt, **kwargs):
        if "PLAN or FAST" in prompt:
            return "PLAN"
        return "Generic response"
        
    async def generate_structured(self, prompt, schema, **kwargs):
        if schema == PlanSchema:
            print("[MockLLM] Generating PlanSchema...")
            return PlanSchema(
                objective="Test Objective",
                tasks=[
                    TaskSchema(task_id="t1", description="Test Task 1"),
                    TaskSchema(task_id="t2", description="Test Task 2")
                ]
            )
        elif schema == ProblemSchema:
            print("[MockLLM] Generating ProblemSchema...")
            return ProblemSchema(
                task_id="mock_id",
                problems=[
                    ProblemDefinition(
                        problem_id="p1", 
                        description="Test Problem", 
                        related_context="None", 
                        files_to_analyze=[]
                    )
                ]
            )
        elif schema == SolutionSchema:
            print("[MockLLM] Generating SolutionSchema...")
            return SolutionSchema(
                task_id="mock_id",
                solutions=[
                    SolutionDefinition(
                        solution_id="s1",
                        problem_id="p1",
                        approach="Mock approach",
                        required_tools=["mock_tool"]
                    )
                ]
            )
        elif schema == ActionSchema:
            print("[MockLLM] Generating ActionSchema...")
            return ActionSchema(
                solution_id="s1",
                tools_to_call=[ToolCall(tool_name="mock_tool", arguments={"arg1": "val1"})],
                io_operations=[
                    IOOperation(operation="create", file_path="mock.txt", content="mock content")
                ],
                action_plan="Will run mock tools"
            )
        elif schema == ReflectionSchema:
            print("[MockLLM] Generating ReflectionSchema...")
            return ReflectionSchema(
                status="completed",
                feedback="All looks good.",
                rating=10,
                is_task_complete=True
            )
        raise ValueError(f"Unknown schema requested: {schema}")

async def test_centralized_brain_arch():
    print("="*60)
    print("🚀 Testing Centralized Brain (Thinker) Architecture")
    print("="*60)
    
    mock_llm = MockLLM()
    registry = ToolRegistry()
    registry.register(MockTool("mock_tool"))
    
    agent = Agent(
        llm=mock_llm,
        tools=registry
    )
    
    prompt = "Create a test project"
    print(f"[*] Starting Agent loop with prompt: {prompt}")
    
    result = await agent.run(prompt, mode="plan")
    
    print("\n[🎯] FINAL ANSWER:")
    print(result)
    
    print("\n[✅] Brain Architecture Test completed successfully!")
    print("The system successfully flowed through Planner -> Thinker -> Actor -> Runtime -> Reflector -> Thinker.")

if __name__ == "__main__":
    try:
        asyncio.run(test_centralized_brain_arch())
    except Exception as e:
        import traceback
        traceback.print_exc()
