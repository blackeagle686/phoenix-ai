import unittest
import asyncio
from unittest.mock import MagicMock, AsyncMock
from phoenix.framework.agent.execution import Actor
from phoenix.framework.agent.execution.tool_manager import ToolManager
from phoenix.framework.agent.tools.base import ToolResult

class TestExecutionFamily(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.mock_registry = MagicMock()
        self.tool_manager = ToolManager(self.mock_registry)
        self.actor = Actor(self.tool_manager)

    async def test_tool_manager_execution(self):
        print("\n--- Testing ToolManager ---")
        mock_tool = MagicMock()
        mock_tool.execute = AsyncMock(return_value=ToolResult(success=True, output="Tool Success"))
        self.mock_registry.get_tool.return_value = mock_tool
        
        print("[*] Executing 'test_tool'...")
        result = await self.tool_manager.execute_tool("test_tool", {"arg": 1})
        print(f"[v] Tool Output: {result}")
        self.assertEqual(result, "Tool Success")
        mock_tool.execute.assert_called_with(arg=1)

    async def test_actor_parallel_execution(self):
        print("\n--- Testing Actor (Parallel) ---")
        mock_tool_1 = MagicMock()
        mock_tool_1.execute = AsyncMock(return_value=ToolResult(success=True, output="Output 1"))
        mock_tool_2 = MagicMock()
        mock_tool_2.execute = AsyncMock(return_value=ToolResult(success=True, output="Output 2"))
        
        def get_tool_mock(name):
            if name == "tool1": return mock_tool_1
            return mock_tool_2
            
        self.mock_registry.get_tool.side_effect = get_tool_mock
        
        plan = {
            "actions": [
                {"tool": "tool1", "kwargs": {}},
                {"tool": "tool2", "kwargs": {}}
            ]
        }
        
        print("[*] Executing multiple tools in parallel...")
        result = await self.actor.execute(plan)
        print(f"[v] Actor Consolidated Result:\n{result}")
        self.assertIn("Output 1", result)
        self.assertIn("Output 2", result)
        self.assertIn("---", result) # Separator check

    async def test_actor_finish_handling(self):
        print("\n--- Testing Actor (Finish) ---")
        plan = {"actions": [{"tool": "finish"}]}
        print("[*] Handling 'finish' action...")
        result = await self.actor.execute(plan)
        print(f"[v] Result: {result}")
        self.assertEqual(result, "Task marked as finished by Planner.")

if __name__ == "__main__":
    unittest.main()

