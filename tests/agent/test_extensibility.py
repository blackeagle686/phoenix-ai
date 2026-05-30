import unittest
from unittest.mock import AsyncMock, MagicMock

from phoenix.framework.agent.core.agent import Agent


class CustomLoop:
    def __init__(self, thinker, planner, actor, reflector, analyzer):
        self.thinker = thinker
        self.planner = planner
        self.actor = actor
        self.reflector = reflector
        self.analyzer = analyzer

    async def run(self, prompt, memory, session_id, max_iterations=5):
        return f"custom-loop:{prompt}"

    async def run_stream(self, prompt, memory, session_id, max_iterations=5):
        yield {"type": "status", "content": "custom loop streaming"}
        yield {"type": "chunk", "content": "custom output\n"}


class TestAgentExtensibility(unittest.IsolatedAsyncioTestCase):
    def _make_base_dependencies(self):
        llm = MagicMock()
        llm.generate = AsyncMock(return_value="FAST")
        llm.client = None
        llm.init = AsyncMock()

        memory = MagicMock()
        memory.add_interaction = AsyncMock()
        memory.get_full_context = AsyncMock(return_value="ctx")

        tools = MagicMock()
        return llm, memory, tools

    async def test_inject_custom_components(self):
        llm, memory, tools = self._make_base_dependencies()
        thinker = MagicMock()
        planner = MagicMock()
        analyzer = MagicMock()
        actor = MagicMock()
        reflector = MagicMock()

        agent = Agent(
            llm=llm,
            memory=memory,
            tools=tools,
            thinker=thinker,
            planner=planner,
            analyzer=analyzer,
            actor=actor,
            reflector=reflector,
            loop_cls=CustomLoop,
        )

        self.assertIs(agent.thinker, thinker)
        self.assertIs(agent.planner, planner)
        self.assertIs(agent.analyzer, analyzer)
        self.assertIs(agent.actor, actor)
        self.assertIs(agent.reflector, reflector)
        self.assertIsInstance(agent.loop, CustomLoop)

    async def test_set_component_and_rebuild_loop(self):
        llm, memory, tools = self._make_base_dependencies()
        agent = Agent(llm=llm, memory=memory, tools=tools, loop_cls=CustomLoop)

        new_planner = MagicMock()
        agent.set_component("planner", new_planner, rebuild_loop=True)
        self.assertIs(agent.planner, new_planner)
        self.assertIs(agent.loop.planner, new_planner)

    async def test_component_factories(self):
        llm, memory, tools = self._make_base_dependencies()
        custom_thinker = MagicMock()
        custom_analyzer = MagicMock()

        factories = {
            "thinker": lambda **kwargs: custom_thinker,
            "analyzer": lambda **kwargs: custom_analyzer,
        }
        agent = Agent(llm=llm, memory=memory, tools=tools, component_factories=factories, loop_cls=CustomLoop)

        self.assertIs(agent.thinker, custom_thinker)
        self.assertIs(agent.analyzer, custom_analyzer)

    async def test_run_uses_custom_loop(self):
        llm, memory, tools = self._make_base_dependencies()
        agent = Agent(llm=llm, memory=memory, tools=tools, loop_cls=CustomLoop)

        result = await agent.run("hello", mode="plan")
        self.assertEqual(result, "custom-loop:hello")

    async def test_run_stream_uses_custom_loop_stream(self):
        llm, memory, tools = self._make_base_dependencies()
        agent = Agent(llm=llm, memory=memory, tools=tools, loop_cls=CustomLoop)

        events = []
        async for event in agent.run_stream("hello", mode="plan"):
            events.append(event)

        self.assertTrue(any(e["type"] == "status" for e in events))
        self.assertTrue(any("custom output" in e["content"] for e in events if e["type"] == "chunk"))


if __name__ == "__main__":
    unittest.main()

