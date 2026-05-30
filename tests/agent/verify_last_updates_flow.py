import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock

# Ensure the project root is in the path so we use the local phoenix source
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from phoenix.framework.agent.core.agent import Agent
from phoenix.framework.agent.core.loop import AgentLoop
from phoenix.framework.agent.memory.hybrid import HybridMemory
from phoenix.framework.agent.memory.adapter import InteractiveMemoryAdapter


class UserCustomMemory:
    def __init__(self):
        self.events = []

    async def add_message(self, session_id, role, content, metadata=None):
        self.events.append((session_id, role, content, metadata or {}))

    async def get_context(self, session_id, query=""):
        return f"[UserCustomMemory] session={session_id}, query={query}"


async def verify_hybrid_memory_flow():
    print("\n=== [1] HybridMemory interactive context flow ===")
    memory = HybridMemory()
    print("[*] Adding interaction with metadata...")
    await memory.add_interaction("flow_s1", "user", "hello phoenix", metadata={"project": "IRYM"})
    print("[*] Adding structured context item...")
    await memory.add_context_item("flow_s1", "repo", "phoenix-ai", source="system")
    print("[*] Building context bundle...")
    bundle = await memory.get_context_bundle("flow_s1", query="hello")
    print(f"[v] Session vars: {bundle['session_variables']}")
    print(f"[v] Full context preview: {bundle['full_text'][:120]}")


async def verify_adapter_flow():
    print("\n=== [2] Custom memory adapter flow ===")
    custom_memory = UserCustomMemory()
    adapter = InteractiveMemoryAdapter(custom_memory)
    print("[*] Adding interaction through adapter...")
    await adapter.add_interaction("flow_s2", "user", "adapter hello", metadata={"src": "verify"})
    print("[*] Fetching full context through adapter...")
    context = await adapter.get_full_context("flow_s2", query="adapter")
    print(f"[v] Adapter context: {context}")
    print(f"[v] Custom backend events: {len(custom_memory.events)}")


async def verify_agent_custom_memory_flow():
    print("\n=== [3] Agent + custom memory integration ===")
    custom_memory = UserCustomMemory()
    llm = MagicMock()
    llm.client = None
    llm.init = AsyncMock()
    llm.generate = AsyncMock(return_value="fast-answer-ok")

    print("[*] Creating Agent with custom memory...")
    agent = Agent(llm=llm, memory=custom_memory, tools=MagicMock())
    print(f"[v] Agent memory wrapper: {agent.memory.__class__.__name__}")

    print("[*] Running in fast mode...")
    result = await agent.run("test custom memory", mode="fast_ans")
    print(f"[v] Agent result: {result}")
    print(f"[v] Stored events in custom memory: {len(custom_memory.events)}")


async def verify_background_reflection_flow():
    print("\n=== [4] Background reflection operation in AgentLoop ===")
    thinker = MagicMock()
    thinker.analyze = AsyncMock(return_value="Objective")
    analyzer = MagicMock()
    analyzer.analyze_workspace = AsyncMock(return_value={"tech_stack": "Python"})
    planner = MagicMock()
    planner.plan = AsyncMock(side_effect=[
        {"actions": [{"tool": "tool1", "kwargs": {}}]},
        {"actions": [{"tool": "finish"}]},
    ])
    planner.stream_thinking = AsyncMock()
    actor = MagicMock()
    actor.execute = AsyncMock(return_value="Action success")
    reflector = MagicMock()
    reflector.llm = MagicMock()
    reflector.reflect = AsyncMock(return_value={"is_complete": False, "reflection": "keep going"})

    memory = MagicMock()
    memory.session = MagicMock()
    memory.reflection = MagicMock()
    memory.add_interaction = AsyncMock()

    async def slow_consolidate(_):
        await asyncio.sleep(0.2)

    memory.consolidate_reflections = AsyncMock(side_effect=slow_consolidate)

    loop = AgentLoop(thinker, planner, actor, reflector, analyzer)
    print("[*] Running loop (reflection consolidation should be backgrounded)...")
    result = await loop.run("do task", memory, "flow_s3", max_iterations=2)
    print(f"[v] Loop result: {result}")
    print(f"[v] Background tasks currently tracked: {len(loop._background_tasks)}")
    await asyncio.sleep(0.25)  # allow background task to complete for visibility
    print(f"[v] Background tasks after wait: {len(loop._background_tasks)}")


async def main():
    print("#############################################")
    print(" Phoenix Last Updates Verification Flow Test ")
    print("#############################################")
    await verify_hybrid_memory_flow()
    await verify_adapter_flow()
    await verify_agent_custom_memory_flow()
    await verify_background_reflection_flow()
    print("\n[✓] Verification flow completed.")


if __name__ == "__main__":
    asyncio.run(main())

