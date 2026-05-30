import unittest
from unittest.mock import AsyncMock, MagicMock

from phoenix.framework.agent.core.agent import Agent
from phoenix.framework.agent.memory.adapter import InteractiveMemoryAdapter
from phoenix.framework.agent.memory.hybrid import HybridMemory


class UserMemoryMinimal:
    """
    Custom memory backend with user-defined method names.
    Adapter should map these into Agent memory contract.
    """

    def __init__(self):
        self.events = []

    async def add_message(self, session_id, role, content, metadata=None):
        self.events.append((session_id, role, content, metadata or {}))

    async def get_context(self, session_id, query=""):
        return f"custom-context for {session_id} query={query}"


class TestCustomMemory(unittest.IsolatedAsyncioTestCase):
    async def test_hybrid_memory_interactive_bundle(self):
        memory = HybridMemory()
        await memory.add_interaction("s1", "user", "hello", metadata={"project": "phoenix"})
        await memory.add_context_item("s1", "repo", "IRYM_sdk", source="system")

        bundle = await memory.get_context_bundle("s1", query="hello")
        self.assertIn("session_variables", bundle)
        self.assertIn("full_text", bundle)
        self.assertEqual(bundle["session_variables"].get("project"), "phoenix")
        self.assertEqual(bundle["session_variables"].get("repo"), "IRYM_sdk")

    async def test_agent_accepts_user_custom_memory_via_adapter(self):
        custom_memory = UserMemoryMinimal()
        llm = MagicMock()
        llm.generate = AsyncMock(return_value="fast-response")
        llm.client = None
        llm.init = AsyncMock()

        agent = Agent(llm=llm, memory=custom_memory, tools=MagicMock())
        self.assertIsInstance(agent.memory, InteractiveMemoryAdapter)

        result = await agent.run("hello", mode="fast_ans")
        self.assertEqual(result, "fast-response")
        self.assertTrue(custom_memory.events)
        self.assertIn("custom-context", llm.generate.call_args_list[-1].args[0])


if __name__ == "__main__":
    unittest.main()

