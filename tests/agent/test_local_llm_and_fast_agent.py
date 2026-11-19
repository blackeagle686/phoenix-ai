import json
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from phoenix.core.hardware_check import HardwareChecker
from phoenix.framework.agent import Agent, LocalLLM, tool
from phoenix.framework.agent.cognition.schema import (
    PlanSchema,
    ReflectionSchema,
    TaskExecutionSchema,
    TaskSchema,
    ToolCallSchema,
)


class TestLocalLLMAndFastAgent(unittest.IsolatedAsyncioTestCase):
    def test_backend_resolution_security_defaults_and_hardware_profile(self):
        hf_llm = LocalLLM(model="Qwen/Qwen2.5-3B-Instruct")
        self.assertEqual(hf_llm._resolve_backend(), "transformers")
        self.assertFalse(hf_llm.trust_remote_code)
        self.assertTrue(hf_llm.is_available())

        gguf_llm = LocalLLM(model="Qwen/Qwen2.5-3B-Instruct-GGUF", gguf_filename="*Q4_K_M.gguf")
        self.assertEqual(gguf_llm._resolve_backend(), "llama_cpp")

        ollama_llm = LocalLLM(model="qwen2.5:3b")
        self.assertEqual(ollama_llm._resolve_backend(), "ollama")

        _, hw = HardwareChecker.check_all(silent=True)
        self.assertIn("recommendation", hw)
        self.assertIn(hw["recommendation"]["device"], ("cuda", "mps", "cpu"))
        self.assertGreaterEqual(hw["cpu"]["recommended_threads"], 1)

    async def test_llama_cpp_gguf_constrained_json_and_streaming(self):
        llm = LocalLLM(model="model-q4_k_m.gguf", backend="llama_cpp")
        llm.active_backend = "llama_cpp"
        llm.active_device = "cpu"
        llm.client = True
        llm._inference_lock = threading.Lock()

        captured_kwargs = []

        def fake_chat_completion(**kwargs):
            captured_kwargs.append(kwargs)
            if kwargs.get("stream"):
                return iter([
                    {"choices": [{"delta": {"content": "Hello "}}]},
                    {"choices": [{"delta": {"content": "ERP!"}}]},
                ])
            if "response_format" in kwargs:
                return {
                    "choices": [{"message": {"content": json.dumps({"objective": "Check stock", "tasks": []})}}],
                    "usage": {"prompt_tokens": 25, "completion_tokens": 12, "total_tokens": 37},
                }
            return {
                "choices": [{"message": {"content": "Stock is 150 units."}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 6, "total_tokens": 16},
            }

        llm.llama_model = SimpleNamespace(create_chat_completion=fake_chat_completion)

        text = await llm.generate("Check stock")
        self.assertEqual(text, "Stock is 150 units.")

        plan = await llm.generate_structured("Plan stock check", PlanSchema)
        self.assertIsInstance(plan, PlanSchema)
        self.assertEqual(plan.objective, "Check stock")
        self.assertEqual(captured_kwargs[-1]["response_format"]["type"], "json_object")

        streamed = []
        async for chunk in llm.generate_stream("Greet ERP"):
            streamed.append(chunk)
        self.assertEqual("".join(streamed), "Hello ERP!")

    async def test_ollama_native_tools_and_fast_react_agent_completes_in_2_calls(self):
        @tool
        def get_customer_balance(customer_id: str) -> dict:
            """Lookup customer balance in ERP."""
            return {"customer_id": customer_id, "balance_usd": 4250.50, "status": "active"}

        llm = LocalLLM(model="qwen2.5:7b", backend="ollama")
        llm.active_backend = "ollama"
        llm.active_device = "cuda"
        llm.is_ollama = True
        llm.client = True
        llm._inference_lock = threading.Lock()

        call_count = 0

        async def fake_post(path, json=None):
            nonlocal call_count
            call_count += 1
            self.assertEqual(path, "/api/chat")
            self.assertEqual(json["keep_alive"], "30m")

            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            if call_count == 1:
                # Native Ollama tool call on step 1
                self.assertIn("tools", json)
                resp.json.return_value = {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {"function": {"name": "get_customer_balance", "arguments": {"customer_id": "CUST-99"}}}
                        ],
                    },
                    "prompt_eval_count": 45,
                    "eval_count": 18,
                }
            else:
                # Direct final answer on step 2 after observing tool output
                resp.json.return_value = {
                    "message": {
                        "role": "assistant",
                        "content": "Customer CUST-99 has an active balance of $4,250.50.",
                    },
                    "prompt_eval_count": 65,
                    "eval_count": 15,
                }
            return resp

        fake_client = MagicMock()
        fake_client.is_closed = False
        fake_client.post = AsyncMock(side_effect=fake_post)
        llm._get_http_client = MagicMock(return_value=fake_client)

        # Agent auto-enables fast_mode=True for LocalLLM!
        agent = Agent(llm=llm, tools=[get_customer_balance])
        self.assertTrue(agent.fast_mode)
        self.assertTrue(agent.loop.fast_reflect)

        answer = await agent.run("What is the balance for customer CUST-99?", session_id="erp_s1", mode="auto")
        self.assertIn("4,250.50", answer)
        # Only 2 LLM calls total (no classifier, no planner, no reflector, no separate synthesizer)!
        self.assertEqual(call_count, 2)

        metrics = agent.get_last_run_metrics()
        self.assertEqual(metrics["llm_calls"], 2)
        self.assertEqual(metrics["tool_calls"], 1)
        self.assertEqual(metrics["total_tokens"], (45 + 18) + (65 + 15))

    async def test_fast_reflect_skips_llm_reflection_on_clean_success_but_runs_on_failure(self):
        @tool
        def create_invoice(amount: float) -> str:
            """Create an ERP invoice."""
            if amount < 0:
                raise ValueError("Negative amount")
            return f"INV-2026:{amount}"

        llm = LocalLLM(model="Qwen/Qwen2.5-3B-Instruct", backend="transformers")
        llm.active_backend = "transformers"
        llm.client = True

        agent = Agent(llm=llm, tools=[create_invoice], fast_mode=True)
        agent.planner.generate_initial_plan = AsyncMock(
            return_value=PlanSchema(
                objective="Create invoice",
                tasks=[TaskSchema(task_id="t1", description="Create $100 invoice")],
            )
        )
        agent.thinker.solve_task = AsyncMock(
            return_value=TaskExecutionSchema(
                thought_process="Create invoice",
                tools_to_call=[ToolCallSchema(tool_name="create_invoice", arguments={"amount": 100.0})],
                is_complete=True,
            )
        )
        agent.reflector.reflect = AsyncMock(
            return_value=ReflectionSchema(is_task_complete=True, rating=9, feedback="ok")
        )
        agent.thinker.synthesize_final_answer = AsyncMock(return_value="Invoice INV-2026:100.0 created.")

        res = await agent.run("Create $100 invoice", session_id="erp_plan_1", mode="plan")
        self.assertIn("INV-2026:100.0", res)
        # Clean tool success -> fast_reflect skipped the LLM Reflector call!
        self.assertEqual(agent.reflector.reflect.call_count, 0)

        # Now simulate a tool failure -> Reflector MUST still run to handle failure!
        agent.thinker.solve_task = AsyncMock(
            return_value=TaskExecutionSchema(
                thought_process="Invalid amount",
                tools_to_call=[ToolCallSchema(tool_name="create_invoice", arguments={"amount": -5.0})],
                is_complete=True,
            )
        )
        agent.reflector.reflect = AsyncMock(
            return_value=ReflectionSchema(is_task_complete=True, rating=3, feedback="Caught negative amount")
        )
        await agent.run("Create -$5 invoice", session_id="erp_plan_2", mode="plan")
        self.assertGreaterEqual(agent.reflector.reflect.call_count, 1)

    async def test_fast_mode_alias_and_greeting_shortcut_take_1_call(self):
        llm = LocalLLM(model="qwen2.5:3b", backend="ollama")
        llm.active_backend = "ollama"
        llm.client = True
        llm.generate = AsyncMock(return_value="Hello! How can I help you?")

        @tool
        def dummy_tool(x: str) -> str:
            """Dummy tool."""
            return x

        agent = Agent(llm=llm, tools=[dummy_tool])

        res1 = await agent.run("hello", mode="fast_mode")
        self.assertEqual(res1, "Hello! How can I help you?")
        self.assertEqual(llm.generate.call_count, 1)

        res2 = await agent.run("hello", mode="auto")
        self.assertEqual(res2, "Hello! How can I help you?")
        self.assertEqual(llm.generate.call_count, 2)


if __name__ == "__main__":
    unittest.main()
