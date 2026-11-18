import json
import os
import tempfile
import unittest

from phoenix.framework.agent import Agent, tool
from phoenix.framework.agent.memory import MemoryManager
from phoenix.framework.agent.runtime.restricted_python import RestrictedPythonRuntime


class RealisticMockLLM:
    """
    Simulates real LLM HTTP responses by returning raw JSON strings (just like
    OpenAI/OpenRouter chat completions) to `generate(...)`, exercising the real
    Thinker, Planner, Actor, Reflector, JSON parser, and MemoryManager end-to-end.
    """

    def __init__(self):
        self.client = None
        self.calls = []
        self.fail_first_write = False
        self._write_attempts = 0

    async def init(self):
        self.client = object()

    async def generate(self, prompt: str, session_id=None, max_tokens=None, **kwargs) -> str:
        self.calls.append({"prompt": prompt, "max_tokens": max_tokens})

        # 1. Auto-mode router classification
        if "Respond with exactly one word: 'PLAN' or 'FAST'." in prompt:
            if "capital of France" in prompt or "quick greeting" in prompt:
                return "FAST"
            return "PLAN"

        # 2. Fast answer mode
        if "Answer directly:" in prompt:
            return "The capital of France is Paris."

        # 3. Thinker.generate_plan (PlanSchema JSON)
        if "You are the master Thinker and Planner." in prompt:
            return json.dumps({
                "objective": "Calculate discounted order total and write summary file",
                "tasks": [
                    {
                        "task_id": "t1",
                        "description": "Calculate 20% discount on 150.0 using calc_discount",
                        "task_type": "other",
                        "priority": "high",
                        "status": "pending",
                        "dependencies": [],
                    },
                    {
                        "task_id": "t2",
                        "description": "Write the discounted price into summary.txt",
                        "task_type": "write",
                        "priority": "high",
                        "status": "pending",
                        "dependencies": ["t1"],
                    },
                ],
            })

        # 4. Thinker.replan (PlanSchema JSON)
        if "performing dynamic re-planning" in prompt:
            return json.dumps({
                "objective": "Recover and write summary.txt",
                "tasks": [
                    {
                        "task_id": "t2_recovery",
                        "description": "Write recovery summary.txt",
                        "task_type": "write",
                        "priority": "high",
                        "status": "pending",
                        "dependencies": [],
                    }
                ],
            })

        # 5. Thinker.solve_task (TaskExecutionSchema JSON)
        if "You are the Thinker. Given the following task" in prompt:
            if "Task ID: t1" in prompt:
                return json.dumps({
                    "thought_process": "Call custom calc_discount tool for price=150.0, rate=20.0",
                    "tools_to_call": [
                        {"tool_name": "calc_discount", "arguments": {"price": 150.0, "rate": 20.0}}
                    ],
                    "is_complete": True,
                })
            if "Task ID: t2_recovery" in prompt:
                return json.dumps({
                    "thought_process": "Write recovery output",
                    "tools_to_call": [
                        {"tool_name": "file_write", "arguments": {"file_path": "summary.txt", "content": "Total=120.0"}}
                    ],
                    "is_complete": True,
                })
            if "Task ID: t2" in prompt:
                if self.fail_first_write and self._write_attempts == 0:
                    self._write_attempts += 1
                    return json.dumps({
                        "thought_process": "Attempt write to invalid tool to trigger retry/replan",
                        "tools_to_call": [
                            {"tool_name": "nonexistent_tool", "arguments": {}}
                        ],
                        "is_complete": False,
                    })
                return json.dumps({
                    "thought_process": "Write Total=120.0 from t1 dependency output into summary.txt",
                    "tools_to_call": [
                        {"tool_name": "file_write", "arguments": {"file_path": "summary.txt", "content": "Total=120.0"}}
                    ],
                    "is_complete": True,
                })

        # 6. Thinker.generate_reflection (ReflectionSchema JSON)
        if "You are the Thinker reflecting on the output" in prompt:
            if "nonexistent_tool" in prompt or "'success': False" in prompt or '"success": false' in prompt.lower():
                return json.dumps({
                    "status": "failed",
                    "feedback": "Tool failed; replanning needed.",
                    "rating": 2,
                    "is_task_complete": False,
                    "needs_replan": True,
                    "summary": "Failed tool call",
                })
            return json.dumps({
                "status": "done",
                "feedback": "Task executed cleanly and verified.",
                "rating": 9,
                "is_task_complete": True,
                "needs_replan": False,
                "summary": "Completed successfully",
            })

        # 7. Thinker.synthesize_final_answer
        if "You are the Phoenix Autonomous Agent. You have finished executing" in prompt:
            return "Discounted total is 120.0 and summary.txt has been written."

        return "OK"


class TestCIE2EMockLLM(unittest.IsolatedAsyncioTestCase):
    async def test_ci_auto_fast_and_plan_e2e_flow(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            llm = RealisticMockLLM()
            mem = MemoryManager(max_context_tokens=1200, stm_window=10)
            rt = RestrictedPythonRuntime(workspace_root=tmpdir)

            @tool
            def calc_discount(price: float, rate: float = 10.0) -> float:
                """Calculate discounted price."""
                return round(price * (1.0 - rate / 100.0), 2)

            agent = Agent(llm=llm, memory=mem, runtime=rt, custom_tools=[calc_discount])

            # 1. Fast mode via auto-router
            fast_res = await agent.run("What is the capital of France?", session_id="ci_s1", mode="auto")
            self.assertEqual(fast_res, "The capital of France is Paris.")

            # 2. Plan mode via auto-router (uses real Thinker JSON parsing, DAG, custom tool, file_write, reflection, synthesis)
            plan_res = await agent.run(
                "Calculate 20% discount on 150 and write it to summary.txt",
                session_id="ci_s1",
                mode="auto",
            )
            self.assertIn("Discounted total is 120.0", plan_res)

            out_file = os.path.join(tmpdir, "summary.txt")
            self.assertTrue(os.path.exists(out_file))
            with open(out_file, "r", encoding="utf-8") as f:
                self.assertEqual(f.read(), "Total=120.0")

            # Verify memory stats (STM + LTM + Working Cells)
            stats = mem.get_stats()
            self.assertEqual(stats["stm_active_turns"], 4)  # 2 fast turns + 2 plan turns
            self.assertGreaterEqual(stats["ltm_items"], 1)

    async def test_ci_streaming_and_replan_recovery(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            llm = RealisticMockLLM()
            llm.fail_first_write = True
            rt = RestrictedPythonRuntime(workspace_root=tmpdir)

            @tool
            def calc_discount(price: float, rate: float = 10.0) -> float:
                """Calculate discounted price."""
                return round(price * (1.0 - rate / 100.0), 2)

            agent = Agent(llm=llm, runtime=rt, custom_tools=[calc_discount])

            events = []
            async for ev in agent.run_stream("Calculate discount and save summary.txt", session_id="ci_s2", mode="plan"):
                events.append(ev)

            self.assertTrue(any(e["type"] == "status" for e in events))
            chunks = "".join(e["content"] for e in events if e["type"] == "chunk")
            self.assertIn("summary.txt", chunks)


if __name__ == "__main__":
    unittest.main()
