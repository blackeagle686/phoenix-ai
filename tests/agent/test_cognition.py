import unittest
import asyncio
from unittest.mock import MagicMock, AsyncMock
from phoenix.framework.agent.cognition.thinker import Thinker
from phoenix.framework.agent.cognition.analyzer import Analyzer
from phoenix.framework.agent.cognition.planner import Planner
from phoenix.framework.agent.cognition.reflector import Reflector
from phoenix.framework.agent.memory.hybrid import HybridMemory

class TestCognitionFamily(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.mock_llm = MagicMock()
        self.mock_llm.generate = AsyncMock()
        self.memory = HybridMemory()

    async def test_thinker_analyze(self):
        print("\n--- Testing Thinker ---")
        thinker = Thinker(llm=self.mock_llm)
        self.mock_llm.generate.return_value = "Core Intent: Test; Requirements: None; Success: Result"
        
        print("[*] Analyzing prompt...")
        result = await thinker.analyze("Do something", self.memory, "session_1")
        print(f"[v] Thinker Output: {result}")
        self.assertIn("Core Intent", result)
        self.mock_llm.generate.assert_called()

    async def test_analyzer_workspace(self):
        print("\n--- Testing Analyzer ---")
        analyzer = Analyzer(llm=self.mock_llm)
        self.mock_llm.generate.return_value = '{"relevant_files": ["main.py"], "tech_stack": "Python", "summary": "Test"}'
        
        print("[*] Analyzing workspace structure...")
        result = await analyzer.analyze_workspace("Find main.py", root_dir=".")
        print(f"[v] Detected Stack: {result['tech_stack']}")
        self.assertEqual(result["tech_stack"], "Python")
        self.assertIn("main.py", result["relevant_files"])

    async def test_planner_plan(self):
        print("\n--- Testing Planner ---")
        mock_tools = MagicMock()
        mock_tools.get_all_tools_info.return_value = []
        planner = Planner(llm=self.mock_llm, tools=mock_tools)
        
        self.mock_llm.generate.return_value = '{"actions": [{"tool": "test_tool", "kwargs": {}}]}'
        
        print("[*] Generating plan...")
        result = await planner.plan("objective", "previous results")
        print(f"[v] Planner Actions: {result['actions']}")
        self.assertIn("actions", result)
        self.assertEqual(result["actions"][0]["tool"], "test_tool")

    async def test_reflector_reflect(self):
        print("\n--- Testing Reflector ---")
        reflector = Reflector(llm=self.mock_llm)
        self.mock_llm.generate.return_value = '{"is_complete": true, "reflection": "Task done"}'
        
        print("[*] Reflecting on outcome...")
        result = await reflector.reflect("objective", {"tool": "act"}, "result")
        print(f"[v] Reflection: {result['reflection']}")
        self.assertTrue(result["is_complete"])
        self.assertEqual(result["reflection"], "Task done")

if __name__ == "__main__":
    unittest.main()

