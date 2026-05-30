import unittest
import asyncio
from unittest.mock import MagicMock, AsyncMock
from phoenix.framework.agent.memory.hybrid import HybridMemory
from phoenix.framework.agent.memory.reflection import ReflectionMemory
from phoenix.framework.agent.memory.session import SessionMemory

class TestMemorySystem(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.memory = HybridMemory()

    async def test_session_memory(self):
        print("\n--- Testing SessionMemory ---")
        print("[*] Setting session variable...")
        self.memory.session.set("key", "value")
        val = self.memory.session.get("key")
        print(f"[v] Retrieved: {val}")
        self.assertEqual(val, "value")

    async def test_reflection_consolidation(self):
        print("\n--- Testing ReflectionConsolidation ---")
        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock(return_value="Consolidated Insight")
        
        print("[*] Adding 5 reflections to trigger threshold...")
        # Add 5 reflections to trigger consolidation threshold
        for i in range(5):
            self.memory.reflection.add_reflection(f"Lesson {i}")
            
        print("[*] Consolidating...")
        await self.memory.consolidate_reflections(mock_llm)
        
        reflections = self.memory.reflection.get_reflections()
        print(f"[v] Consolidated Reflections: {reflections}")
        self.assertIn("Consolidated Insight", reflections)
        self.assertEqual(len(self.memory.reflection.reflections), 1)

    async def test_parallel_context_retrieval(self):
        print("\n--- Testing ParallelContextRetrieval ---")
        # This test ensures the parallel retrieval logic doesn't crash
        print("[*] Retrieving full context in parallel...")
        context = await self.memory.get_full_context("session_1", query="test")
        print("[v] Parallel retrieval successful.")
        self.assertIsInstance(context, str)

if __name__ == "__main__":
    unittest.main()

