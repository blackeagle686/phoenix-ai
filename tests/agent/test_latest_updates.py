import os
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock

from phoenix.framework.agent.core.loop import AgentLoop
from phoenix.framework.agent.cognition.planner import Planner
from phoenix.framework.agent.tools.io import FileAppendTool, FileEditTool


class TestLatestUpdates(unittest.IsolatedAsyncioTestCase):
    async def test_planner_stream_thinking_fallback(self):
        mock_llm = MagicMock()
        # Disable implicit MagicMock attribute auto-creation for generate_stream.
        # This forces the Planner to exercise the non-stream fallback path.
        mock_llm.generate_stream = None
        mock_llm.generate = AsyncMock(return_value="I will read then patch the file.")
        mock_tools = MagicMock()
        mock_tools.get_all_tools_info.return_value = []
        planner = Planner(llm=mock_llm, tools=mock_tools)

        chunks = []
        async for chunk in planner.stream_thinking("Update config safely", "none"):
            chunks.append(chunk)

        joined = "".join(chunks)
        self.assertIn("read then patch", joined)
        mock_llm.generate.assert_called_once()

    async def test_agent_loop_stream_emits_planner_thoughts(self):
        thinker = MagicMock()
        thinker.analyze = AsyncMock(return_value="Objective")

        analyzer = MagicMock()
        analyzer.analyze_workspace = AsyncMock(return_value={"tech_stack": "Python"})

        planner = MagicMock()

        async def fake_stream_thinking(objective, *args, **kwargs):
            yield "Planner thought line 1. "
            yield "Planner thought line 2."

        planner.stream_thinking = fake_stream_thinking
        planner.plan = AsyncMock(
            side_effect=[
                {"actions": [{"tool": "tool1", "kwargs": {}}]},
                {"actions": [{"tool": "finish"}]},
            ]
        )

        actor = MagicMock()
        actor.execute = AsyncMock(return_value="Action Success")

        reflector = MagicMock()
        reflector.llm = MagicMock()
        reflector.reflect = AsyncMock(return_value={"is_complete": False, "reflection": "Continue"})

        loop = AgentLoop(thinker, planner, actor, reflector, analyzer)

        memory = MagicMock()
        memory.session = MagicMock()
        memory.reflection = MagicMock()
        memory.add_interaction = AsyncMock()
        memory.consolidate_reflections = AsyncMock()

        events = []
        async for event in loop.run_stream("test prompt", memory, "s1", max_iterations=2):
            events.append(event)

        statuses = [e["content"] for e in events if e["type"] == "status"]
        chunks = "".join([e["content"] for e in events if e["type"] == "chunk"])
        self.assertTrue(any("Planner thinking" in s for s in statuses))
        self.assertIn("Planner thought line 1.", chunks)
        self.assertIn("Planner thought line 2.", chunks)
        self.assertIn("Action Success", chunks)

    async def test_agent_loop_prevents_finish_without_actions(self):
        thinker = MagicMock()
        thinker.analyze = AsyncMock(return_value="Objective")

        analyzer = MagicMock()
        analyzer.analyze_workspace = AsyncMock(return_value={"tech_stack": "Python"})

        planner = MagicMock()
        planner.plan = AsyncMock(
            side_effect=[
                {"actions": [{"tool": "finish"}]},
                {"actions": [{"tool": "tool1", "kwargs": {}}]},
                {"actions": [{"tool": "finish"}]},
            ]
        )

        actor = MagicMock()
        actor.execute = AsyncMock(return_value="Action Success")

        reflector = MagicMock()
        reflector.llm = MagicMock()
        reflector.reflect = AsyncMock(return_value={"is_complete": False, "reflection": "More to do"})

        loop = AgentLoop(thinker, planner, actor, reflector, analyzer)

        memory = MagicMock()
        memory.session = MagicMock()
        memory.reflection = MagicMock()
        memory.add_interaction = AsyncMock()
        memory.consolidate_reflections = AsyncMock()

        result = await loop.run("prompt", memory, "s2", max_iterations=3)
        self.assertIn("Action Success", result)
        self.assertEqual(actor.execute.call_count, 3)

    async def test_file_append_and_file_edit_upsert(self):
        append_tool = FileAppendTool()
        edit_tool = FileEditTool()

        with tempfile.TemporaryDirectory() as tmp:
            file_path = os.path.join(tmp, "sample.py")

            res1 = await append_tool.execute(file_path=file_path, content="print('hello')\n")
            self.assertTrue(res1.success)

            res2 = await edit_tool.execute(
                file_path=file_path,
                edits=[{"search": "print('hello')", "replace": "print('hi')"}],
                upsert=True,
            )
            self.assertTrue(res2.success)

            res3 = await edit_tool.execute(
                file_path=file_path,
                edits=[{"search": "def missing():", "replace": "def missing():\n    return 1"}],
                upsert=True,
            )
            self.assertTrue(res3.success)

            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            self.assertIn("print('hi')", content)
            self.assertIn("def missing():", content)


if __name__ == "__main__":
    unittest.main()
