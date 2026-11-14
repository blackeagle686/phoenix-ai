import asyncio
import os
import tempfile
import unittest
from unittest.mock import AsyncMock

from phoenix.framework.agent.core.agent import Agent
from phoenix.framework.agent.core.profile import AgentProfile
from phoenix.framework.agent.cognition.schema import (
    PlanSchema,
    TaskSchema,
    TaskExecutionSchema,
    ToolCallSchema,
    ReflectionSchema,
)
from phoenix.framework.agent.tools.base import BaseTool, ToolResult
from phoenix.framework.agent.tools.registry import ToolRegistry
from phoenix.framework.agent.execution.tool_manager import ToolManager
from phoenix.framework.agent.runtime.restricted_python import RestrictedPythonRuntime


class SampleTool(BaseTool):
    name = "sample_calc"
    description = "Calculates a weighted score."

    async def execute(self, count: int, ratio: float = 0.5, label: str = "default") -> ToolResult:
        return ToolResult(success=True, output=f"{label}:{count * ratio}")


class TestV2ArchitectureUpgrade(unittest.IsolatedAsyncioTestCase):
    async def test_tool_parameter_schema_introspection_and_profile_guard(self):
        tool = SampleTool()
        schema = tool.get_parameters_schema()
        self.assertEqual(schema["type"], "object")
        self.assertIn("count", schema["properties"])
        self.assertEqual(schema["properties"]["count"]["type"], "integer")
        self.assertEqual(schema["properties"]["ratio"]["type"], "number")
        self.assertIn("count", schema["required"])
        self.assertNotIn("ratio", schema["required"])

        reg = ToolRegistry()
        reg.register(tool)
        profile = AgentProfile.from_dict({
            "identity": {"name": "Restricted", "id": "r1"},
            "role": {"title": "Tester", "mission": "Test"},
            "personality": {"communication_tone": "direct", "response_style": "concise"},
            "tool_access": ["other_tool"],
        })
        tm = ToolManager(reg, profile=profile)
        denied = await tm.execute_tool_result("sample_calc", count=4)
        self.assertFalse(denied.success)
        self.assertIn("not permitted", denied.error)

        allowed_profile = AgentProfile.from_dict({
            "identity": {"name": "Allowed", "id": "a1"},
            "role": {"title": "Tester", "mission": "Test"},
            "personality": {"communication_tone": "direct", "response_style": "concise"},
            "tool_access": ["sample_calc"],
        })
        tm_allowed = ToolManager(reg, profile=allowed_profile)
        hook_events = []
        tm_allowed.register_before_hook(lambda name, kwargs: hook_events.append(("before", name)))
        tm_allowed.register_after_hook(lambda name, kwargs, res: hook_events.append(("after", name, res.success)))
        ok = await tm_allowed.execute_tool_result("sample_calc", count=4, ratio=1.5, label="score")
        self.assertTrue(ok.success)
        self.assertEqual(ok.output, "score:6.0")
        self.assertEqual(hook_events, [("before", "sample_calc"), ("after", "sample_calc", True)])

    async def test_restricted_runtime_workspace_jail_and_dunder_block(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rt = RestrictedPythonRuntime(workspace_root=tmpdir, timeout=5.0)
            outside_res = await rt.execute_io("read", "/etc/passwd")
            self.assertFalse(outside_res.success)
            self.assertIn("outside workspace root", outside_res.error)

            escape_code = "x = ().__class__.__base__.__subclasses__()"
            code_res = await rt.execute_code(escape_code)
            self.assertFalse(code_res.success)
            self.assertIn("blocked for security reasons", code_res.error)

    async def test_parallel_dag_intra_task_react_and_final_synthesis(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            src_path = os.path.join(tmpdir, "input.txt")
            out_path = os.path.join(tmpdir, "output.txt")
            with open(src_path, "w", encoding="utf-8") as f:
                f.write("SECRET_VALUE_42")

            rt = RestrictedPythonRuntime(workspace_root=tmpdir)
            agent = Agent(runtime=rt)

            # Build a 3-task DAG: t1 and t2 have no dependencies, t3 depends on t1 and t2
            plan = PlanSchema(
                objective="Read input and write transformed output",
                tasks=[
                    TaskSchema(
                        task_id="t1",
                        description="Inspect input.txt",
                        task_type="read",
                        priority="high",
                        dependencies=[],
                    ),
                    TaskSchema(
                        task_id="t2",
                        description="Compute metadata",
                        task_type="other",
                        priority="medium",
                        dependencies=[],
                    ),
                    TaskSchema(
                        task_id="t3",
                        description="Write final output.txt using t1 and t2",
                        task_type="write",
                        priority="high",
                        dependencies=["t1", "t2"],
                    ),
                ],
            )
            agent.planner.generate_initial_plan = AsyncMock(return_value=plan)

            # Configure thinker.solve_task to test intra-task ReAct follow-up on t1:
            # First call on t1 runs file_read; follow-up call on t1 produces direct_response with the read value.
            t1_calls = 0

            async def fake_solve_task(task, context=""):
                nonlocal t1_calls
                if task.task_id == "t1":
                    t1_calls += 1
                    if t1_calls == 1:
                        return TaskExecutionSchema(
                            thought_process="Read input.txt first",
                            tools_to_call=[ToolCallSchema(tool_name="file_read", arguments={"file_path": src_path})],
                            is_complete=False,
                        )
                    return TaskExecutionSchema(
                        thought_process="Observed SECRET_VALUE_42 from file_read",
                        direct_response="Found SECRET_VALUE_42",
                        is_complete=True,
                    )
                if task.task_id == "t2":
                    return TaskExecutionSchema(
                        thought_process="Run python calculation",
                        tools_to_call=[ToolCallSchema(tool_name="python_repl", arguments={"code": "print(21 * 2)"})],
                        is_complete=True,
                    )
                if task.task_id == "t3":
                    return TaskExecutionSchema(
                        thought_process="Write combined output file",
                        tools_to_call=[
                            ToolCallSchema(
                                tool_name="file_write",
                                arguments={"file_path": out_path, "content": "SECRET_VALUE_42:42"},
                            )
                        ],
                        is_complete=True,
                    )
                raise ValueError(f"Unexpected task {task.task_id}")

            agent.thinker.solve_task = AsyncMock(side_effect=fake_solve_task)
            agent.thinker.generate_reflection = AsyncMock(
                return_value=ReflectionSchema(
                    is_task_complete=True,
                    rating=9,
                    feedback="Completed cleanly",
                    summary="Success",
                )
            )
            agent.thinker.synthesize_final_answer = AsyncMock(
                return_value="Processed input.txt and wrote SECRET_VALUE_42:42 to output.txt."
            )

            final_response = await agent.run("Process input.txt into output.txt", session_id="v2_sess", mode="plan")
            self.assertIn("Processed input.txt and wrote SECRET_VALUE_42:42 to output.txt.", final_response)
            self.assertEqual(t1_calls, 2)
            self.assertTrue(os.path.exists(out_path))
            with open(out_path, "r", encoding="utf-8") as f:
                self.assertEqual(f.read(), "SECRET_VALUE_42:42")

            # Verify unified memory graph populated cells on agent.memory
            self.assertIn("t1", agent.memory.completed_tasks)
            self.assertIn("t2", agent.memory.completed_tasks)
            self.assertIn("t3", agent.memory.completed_tasks)
            t1_ctx = agent.memory.get_cell("t1").to_context_string()
            self.assertIn("SECRET_VALUE_42", t1_ctx)


if __name__ == "__main__":
    unittest.main()
