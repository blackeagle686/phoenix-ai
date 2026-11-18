import os
import tempfile
import unittest
from unittest.mock import AsyncMock

from phoenix.framework.agent import Agent
from phoenix.framework.agent.cognition.schema import (
    PlanSchema,
    TaskSchema,
    TaskExecutionSchema,
    ToolCallSchema,
    ReflectionSchema,
)
from phoenix.framework.agent.memory import (
    MemoryManager,
    MemoryCell,
    ShortTermMemoryManager,
    LongMemoryCell,
    LongTermMemoryManager,
    estimate_tokens,
)


class TestOptimizedContextMemory(unittest.IsolatedAsyncioTestCase):
    async def test_stm_sliding_window_and_rolling_summary(self):
        stm = ShortTermMemoryManager(max_cells=3)
        for i in range(1, 6):
            await stm.add("s1", {"role": "user" if i % 2 else "assistant", "content": f"message number {i}"})

        cells = await stm.get("s1")
        self.assertEqual(len(cells), 3)
        self.assertEqual([c.content for c in cells], ["message number 3", "message number 4", "message number 5"])
        self.assertEqual(stm.total_evicted("s1"), 2)

        ctx = await stm.get_context_string("s1")
        self.assertIn("[Earlier Summary]", ctx)
        self.assertIn("message number 1", ctx)
        self.assertIn("message number 5", ctx)

    def test_ltm_dedup_relevance_retrieval_and_jsonl_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "ltm.jsonl")
            ltm = LongTermMemoryManager(max_items=50, storage_path=db_path)

            c1 = ltm.add("User prefers concise Arabic explanations with Python examples.", memory_type="preference")
            c2 = ltm.add("User prefers concise Arabic explanations with Python examples.", memory_type="preference")
            self.assertEqual(c1.memory_id, c2.memory_id)
            self.assertEqual(ltm.count(), 1)

            ltm.add("Production API uses FastAPI on port 8000 with Redis caching.", memory_type="fact")
            ltm.add("Docker image build requires Python 3.12 base image.", memory_type="fact")

            results = ltm.retrieve("Arabic Python preferences", limit=2)
            self.assertTrue(results)
            self.assertIn("Arabic", results[0].content)

            # Reload from disk and verify persistence
            ltm_reloaded = LongTermMemoryManager(max_items=50, storage_path=db_path)
            self.assertEqual(ltm_reloaded.count(), 3)

    async def test_dependency_aware_context_and_strict_token_budget(self):
        mem = MemoryManager(max_context_tokens=300, stm_window=6)
        mem.set_session("s_dep", "Build deployment report")

        # Filter noisy internal system telemetry from STM
        await mem.add_interaction("s_dep", "user", "Check config and build report")
        await mem.add_interaction("s_dep", "system", "Task t1 attempt 1. Success: True")
        stm_turns = await mem.short_term.get("s_dep")
        self.assertEqual(len(stm_turns), 1)
        self.assertEqual(stm_turns[0].role, "user")

        mem.create_cell("t1", "Read config.json", dependencies=[])
        c1 = mem.get_cell("t1")
        c1.add_tool_record("file_read", {"file_path": "config.json"}, True, output={"port": 9090, "env": "prod"})
        mem.mark_complete("t1")

        mem.create_cell("t2", "Fetch unrelated metrics", dependencies=[])
        c2 = mem.get_cell("t2")
        c2.add_tool_record("web_search", {"query": "huge"}, True, output="UNRELATED_NOISE " * 150)
        mem.mark_complete("t2")

        mem.create_cell("t3", "Generate final report from config", dependencies=["t1"])

        task_ctx = mem.get_task_context("t3", max_tokens=280)
        # Critical OBJECTIVE + CURRENT TASK header is never chopped from the front!
        self.assertTrue(task_ctx.startswith("OBJECTIVE:\nBuild deployment report"))
        self.assertIn("CURRENT TASK CONTEXT:", task_ctx)
        # Direct dependency t1 output is included; unrelated huge noise from t2 is NOT dumped!
        self.assertIn("DEPENDENCY OUTPUTS:", task_ctx)
        self.assertIn("9090", task_ctx)
        self.assertNotIn("UNRELATED_NOISE", task_ctx)
        # Fits tightly within token budget (allowing small separator margin)
        self.assertLessEqual(estimate_tokens(task_ctx), 300)

    def test_working_cell_compaction_and_non_redundant_retry_context(self):
        mem = MemoryManager(max_context_tokens=400)
        mem.set_session("s_retry", "Fix bug in handler")
        cell = mem.create_cell("t1", "Run tests and fix")
        cell.increment_attempt()
        huge_file = {"file_path": "/app/main.py", "total_lines": 500, "content": [{"from_line": 1, "to_line": 50, "content_block": "def f(): pass\n" * 80, "token_count": None, "vector_id": None}]}
        cell.add_tool_record("file_read", {"file_path": "/app/main.py"}, True, output=huge_file)
        cell.add_runtime_result({"task_id": "t1", "success": True, "result": {"execution_results": [{"huge": "blob" * 500}], "direct_response": "Read main.py"}})

        retry_ctx = mem.build_retry_context("t1")
        # Metadata boilerplate like token_count: None should be stripped by smart compactor
        self.assertNotIn("token_count", retry_ctx)
        self.assertEqual(retry_ctx.count("Recent Tools:"), 1)

        cell.task_status = "done"
        cell.compact()
        self.assertTrue(all(len(str(r.get("output") or "")) <= 400 for r in cell.tool_history))

    async def test_agent_run_finalizes_memory_and_keeps_footprint_lean(self):
        mem = MemoryManager(max_context_tokens=1000, stm_window=8)

        class MockLLM:
            async def generate(self, prompt: str, **kwargs) -> str:
                return "Final synthesized summary: Order verified."

            async def generate_structured(self, prompt: str, schema_cls, **kwargs):
                if schema_cls is PlanSchema:
                    return PlanSchema(
                        objective="Verify order details",
                        tasks=[TaskSchema(task_id="t1", description="Lookup order 42")],
                    )
                if schema_cls is TaskExecutionSchema:
                    return TaskExecutionSchema(
                        thought_process="Use lookup",
                        tools_to_call=[ToolCallSchema(tool_name="order_lookup", arguments={"order_id": 42})],
                        is_complete=True,
                    )
                if schema_cls is ReflectionSchema:
                    return ReflectionSchema(is_task_complete=True, rating=9, feedback="Verified order 42.")
                raise ValueError(schema_cls)

        def order_lookup(order_id: int) -> dict:
            """Lookup order by ID."""
            return {"order_id": order_id, "status": "delivered"}

        agent = Agent(llm=MockLLM(), memory=mem, custom_tools=[order_lookup])
        res = await agent.run("Verify order 42", session_id="opt_sess", mode="plan")
        self.assertIn("Order verified", res)

        stats = mem.get_stats()
        self.assertEqual(stats["stm_active_turns"], 2)  # Only User prompt + Final Assistant response
        self.assertGreaterEqual(stats["ltm_items"], 1)  # Run outcome selectively promoted to LTM
        self.assertEqual(stats["working_cells"], 1)


if __name__ == "__main__":
    unittest.main()
