import unittest
from unittest.mock import AsyncMock, MagicMock

from phoenix.framework.agent.core.agent import Agent


class UserCustomLLM:
    """
    Minimal user-defined model implementation for framework integration tests.
    """

    def __init__(self, classify_as="FAST"):
        self.classify_as = classify_as
        self.client = None
        self.init = AsyncMock()
        self.generate = AsyncMock(side_effect=self._generate_side_effect)

    async def _generate_side_effect(self, prompt, session_id=None, max_tokens=None):
        if "Respond with exactly one word: 'PLAN' or 'FAST'." in prompt:
            return self.classify_as
        return "custom-model-response"


class CustomLoop:
    async def run(self, prompt, memory, session_id, max_iterations=5):
        return "custom-loop-plan-result"

    async def run_stream(self, prompt, memory, session_id, max_iterations=5):
        yield {"type": "status", "content": "custom loop stream"}
        yield {"type": "chunk", "content": "custom streamed result\n"}


class TestCustomUserModel(unittest.IsolatedAsyncioTestCase):
    def _memory_mock(self):
        memory = MagicMock()
        memory.add_interaction = AsyncMock()
        memory.get_full_context = AsyncMock(return_value="memory context")
        return memory

    async def test_user_custom_model_fast_mode(self):
        llm = UserCustomLLM(classify_as="FAST")
        memory = self._memory_mock()
        tools = MagicMock()

        agent = Agent(llm=llm, memory=memory, tools=tools)
        result = await agent.run("hello", mode="auto")

        self.assertEqual(result, "custom-model-response")
        self.assertGreaterEqual(llm.generate.call_count, 2)

        # Ensure classification call uses the low token budget.
        first_call = llm.generate.call_args_list[0]
        self.assertEqual(first_call.kwargs.get("max_tokens"), 10)

    async def test_user_custom_model_plan_mode_with_custom_loop(self):
        llm = UserCustomLLM(classify_as="PLAN")
        memory = self._memory_mock()
        tools = MagicMock()

        agent = Agent(llm=llm, memory=memory, tools=tools, loop=CustomLoop())
        result = await agent.run("build workflow", mode="plan")

        self.assertEqual(result, "custom-loop-plan-result")

    async def test_user_custom_model_streaming_path(self):
        llm = UserCustomLLM(classify_as="PLAN")
        memory = self._memory_mock()
        tools = MagicMock()

        agent = Agent(llm=llm, memory=memory, tools=tools, loop=CustomLoop())
        events = []
        async for event in agent.run_stream("stream this", mode="plan"):
            events.append(event)

        self.assertTrue(any(e["type"] == "status" for e in events))
        self.assertTrue(any("custom streamed result" in e["content"] for e in events if e["type"] == "chunk"))


if __name__ == "__main__":
    unittest.main()

