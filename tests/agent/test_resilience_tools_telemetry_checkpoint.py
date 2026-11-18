import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from phoenix.framework.agent import Agent, LLMRouter, tool
from phoenix.framework.agent.memory import MemoryManager
from phoenix.services.llm.openai import OpenAILLM


def _fake_chat_response(content: str = "", tool_calls=None, prompt_tokens=10, completion_tokens=5):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=msg)
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    return SimpleNamespace(choices=[choice], usage=usage)


class TestResilienceToolsTelemetryCheckpoint(unittest.IsolatedAsyncioTestCase):
    async def test_single_key_multi_model_fallback_on_429_and_cooldown(self):
        llm = OpenAILLM(
            api_key="or-single-key",
            base_url="https://openrouter.ai/api/v1",
            model="model-primary:free",
            fallback_models=["model-backup-1:free", "model-backup-2:free"],
            retry_base_delay=0.01,
            cooldown_seconds=30.0,
        )
        calls = []

        async def fake_create(**kwargs):
            m = kwargs["model"]
            calls.append(m)
            if m == "model-primary:free":
                raise RuntimeError("Error code: 429 - Too Many Requests")
            return _fake_chat_response(f"ok from {m}", prompt_tokens=12, completion_tokens=8)

        llm.client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock(side_effect=fake_create)))
        )

        # First call hits 429 on primary -> rotates to model-backup-1
        r1 = await llm.generate("hello")
        self.assertEqual(r1, "ok from model-backup-1:free")
        # Second call skips cooled primary immediately -> calls model-backup-1 directly
        r2 = await llm.generate("second call")
        self.assertEqual(r2, "ok from model-backup-1:free")
        self.assertEqual(calls, ["model-primary:free", "model-backup-1:free", "model-backup-1:free"])

    async def test_multi_platform_multi_key_router_with_agent(self):
        class BrokenGroqLLM:
            model = "llama-broken"
            client = True

            async def init(self):
                pass

            async def generate(self, prompt, **kwargs):
                raise RuntimeError("Error code: 429 - Rate limit reached on key 1")

            async def generate_structured(self, prompt, schema, **kwargs):
                raise RuntimeError("Error code: 429 - Rate limit reached on key 1")

        class HealthyOpenRouterLLM:
            model = "openrouter/free"
            client = True

            async def init(self):
                pass

            async def generate(self, prompt, **kwargs):
                if "Respond with exactly one word: 'PLAN' or 'FAST'." in prompt:
                    return "FAST"
                return "Served by fallback key 2!"

        events = []
        agent = Agent(
            llms=[BrokenGroqLLM(), HealthyOpenRouterLLM()],
            on_event=lambda ev: events.append(ev),
        )
        self.assertIsInstance(agent.llm, LLMRouter)

        res = await agent.run("Say hi", mode="auto")
        self.assertEqual(res, "Served by fallback key 2!")

        metrics = agent.get_last_run_metrics()
        self.assertIsNotNone(metrics)
        self.assertGreaterEqual(metrics["failovers"], 1)
        self.assertEqual(metrics["status"], "success")

    async def test_native_tool_calling_and_run_metrics(self):
        @tool
        def multiply_numbers(a: int, b: int) -> int:
            """Multiply two integers."""
            return a * b

        llm = OpenAILLM(api_key="test-key", model="gpt-4o-mini")
        create_calls = []

        async def fake_create(**kwargs):
            create_calls.append(kwargs)
            messages = kwargs["messages"]
            sys_msg = messages[0]["content"] if len(messages) > 1 else ""
            user_msg = messages[-1]["content"]

            if "tools" in kwargs:
                # Native OpenAI function calling path!
                tc = SimpleNamespace(
                    id="call_1",
                    type="function",
                    function=SimpleNamespace(name="multiply_numbers", arguments=json.dumps({"a": 6, "b": 7})),
                )
                return _fake_chat_response("", tool_calls=[tc], prompt_tokens=30, completion_tokens=15)

            if "strict data-generation assistant" in sys_msg and "PlanSchema" in sys_msg:
                return _fake_chat_response(
                    json.dumps({
                        "objective": "Multiply 6 by 7",
                        "tasks": [{"task_id": "t1", "description": "Multiply 6 and 7", "task_type": "other", "priority": "high"}],
                    }),
                    prompt_tokens=40,
                    completion_tokens=20,
                )

            if "strict data-generation assistant" in sys_msg and "ReflectionSchema" in sys_msg:
                return _fake_chat_response(
                    json.dumps({
                        "status": "done",
                        "feedback": "42 returned cleanly",
                        "rating": 10,
                        "is_task_complete": True,
                    }),
                    prompt_tokens=25,
                    completion_tokens=10,
                )

            if "You are the Phoenix Autonomous Agent" in user_msg:
                return _fake_chat_response("6 * 7 = 42.", prompt_tokens=20, completion_tokens=8)

            return _fake_chat_response("FAST", prompt_tokens=5, completion_tokens=2)

        llm.client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock(side_effect=fake_create)))
        )

        agent = Agent(llm=llm, tools=[multiply_numbers])
        ans = await agent.run("Multiply 6 and 7", session_id="native_tc_s1", mode="plan")
        self.assertIn("42", ans)

        # Verify native `tools` kwarg was sent to OpenAI Chat Completions API
        native_calls = [c for c in create_calls if "tools" in c]
        self.assertTrue(len(native_calls) >= 1)
        tool_names = [t["function"]["name"] for t in native_calls[0]["tools"]]
        self.assertIn("multiply_numbers", tool_names)

        # Verify telemetry captured exact tokens + tool call count
        metrics = agent.get_last_run_metrics()
        self.assertEqual(metrics["tool_calls"], 1)
        self.assertEqual(metrics["tool_failures"], 0)
        self.assertGreater(metrics["total_tokens"], 0)
        self.assertIn("gpt-4o-mini", metrics["models_used"])

    async def test_checkpoint_and_resume_skips_completed_tasks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            executed_tasks = []

            @tool
            def record_step(step_name: str) -> str:
                """Records execution of a named step."""
                executed_tasks.append(step_name)
                if step_name == "step_2" and executed_tasks.count("step_2") == 1:
                    raise RuntimeError("Simulated interruption on step_2")
                return f"done:{step_name}"

            class InterruptThenResumeLLM:
                async def generate(self, prompt: str, **kwargs) -> str:
                    return f"Completed steps: {executed_tasks}"

                async def generate_structured(self, prompt: str, schema_cls, **kwargs):
                    from phoenix.framework.agent.cognition.schema import (
                        PlanSchema, TaskSchema, TaskExecutionSchema, ToolCallSchema, ReflectionSchema
                    )
                    if schema_cls is PlanSchema:
                        return PlanSchema(
                            objective="Run two sequential steps",
                            tasks=[
                                TaskSchema(task_id="t1", description="Run step_1", dependencies=[]),
                                TaskSchema(task_id="t2", description="Run step_2", dependencies=["t1"]),
                            ],
                        )
                    if schema_cls is TaskExecutionSchema:
                        step = "step_1" if "Task ID: t1" in prompt else "step_2"
                        return TaskExecutionSchema(
                            thought_process=f"Execute {step}",
                            tools_to_call=[ToolCallSchema(tool_name="record_step", arguments={"step_name": step})],
                            is_complete=True,
                        )
                    if schema_cls is ReflectionSchema:
                        ok = "Simulated interruption" not in prompt
                        return ReflectionSchema(
                            is_task_complete=ok,
                            rating=9 if ok else 2,
                            feedback="OK" if ok else "Interrupted",
                        )
                    raise ValueError(schema_cls)

            # First agent fails/stops cleanly after t1 completes while checkpointing each wave
            agent1 = Agent(
                llm=InterruptThenResumeLLM(),
                memory=MemoryManager(),
                tools=[record_step],
                checkpoint_dir=tmpdir,
            )
            # Human rejects t2 after t1 succeeds -> simulates mid-run pause/stop with checkpoint saved!
            agent1.loop.approval_callback = lambda task, ctx: "reject" if task.task_id == "t2" else True

            paused_res = await agent1.run("Run two steps", session_id="ckpt_s1", mode="plan")
            self.assertIn("rejection", paused_res.lower())
            self.assertEqual(executed_tasks, ["step_1"])

            cp_file = os.path.join(tmpdir, "ckpt_s1.json")
            self.assertTrue(os.path.exists(cp_file))

            # New Agent instance resumes the same session from checkpoint_dir: t1 MUST NOT re-run!
            executed_tasks.clear()
            agent2 = Agent(
                llm=InterruptThenResumeLLM(),
                memory=MemoryManager(),
                tools=[record_step],
                checkpoint_dir=tmpdir,
            )
            # Clear simulated crash condition by overriding step_2 count: only step_2 runs now!
            resumed_res = await agent2.resume("ckpt_s1", max_iterations=5)
            self.assertNotIn("step_1", executed_tasks)
            self.assertIn("step_2", executed_tasks)
            self.assertIn("Completed steps:", resumed_res)


if __name__ == "__main__":
    unittest.main()
