import unittest
import asyncio
from unittest.mock import MagicMock, AsyncMock
from phoenix.framework.agent.core.loop import AgentLoop
from phoenix.framework.agent.cognition.pipeline import CognitionPipeline

class TestAgentLoop(unittest.IsolatedAsyncioTestCase):
    async def test_loop_run(self):
        print("\n--- Testing AgentLoop Integration ---")
        # Mocking all dependencies
        print("[*] Setting up mocks for Thinker, Analyzer, Planner, Actor, Reflector...")
        thinker = MagicMock()
        thinker.analyze = AsyncMock(return_value="Objective")
        
        analyzer = MagicMock()
        analyzer.analyze_workspace = AsyncMock(return_value={"tech_stack": "Python"})
        
        planner = MagicMock()
        # First action then finish
        planner.plan = AsyncMock(side_effect=[
            {"actions": [{"tool": "tool1", "kwargs": {}}]},
            {"actions": [{"tool": "finish"}]}
        ])
        
        actor = MagicMock()
        actor.execute = AsyncMock(return_value="Action Success")
        
        reflector = MagicMock()
        reflector.reflect = AsyncMock(return_value={"is_complete": False, "reflection": "More to do"})
        
        loop = AgentLoop(thinker, planner, actor, reflector, analyzer)
        
        # Configure bootstrap pipeline to include thinker and analyzer for testing custom setup
        loop._bootstrap_pipeline = CognitionPipeline({
            "name": "test_bootstrap",
            "version": "1.0",
            "steps": [
                {
                    "id": "thinker_step",
                    "brain": "thinker",
                    "required_inputs": ["prompt", "memory", "session_id"],
                    "input_map": {
                        "prompt": "prompt",
                        "memory": "memory",
                        "session_id": "session_id"
                    },
                    "output_key": "thinker_output"
                },
                {
                    "id": "analyzer_step",
                    "brain": "analyzer",
                    "required_inputs": ["prompt"],
                    "input_map": {
                        "prompt": "prompt"
                    },
                    "output_key": "analyzer_output"
                }
            ]
        })
        
        memory = MagicMock()
        memory.session = MagicMock()
        memory.reflection = MagicMock()
        memory.add_interaction = AsyncMock()
        memory.consolidate_reflections = AsyncMock()

        print("[*] Running AgentLoop for 2 iterations...")
        result = await loop.run("test prompt", memory, "session_1", max_iterations=2)
        
        print(f"[v] Loop Result: {result}")
        self.assertIn("Action Success", result)
        
        print("[*] Verifying component calls...")
        thinker.analyze.assert_called_once()
        analyzer.analyze_workspace.assert_called_once()
        self.assertEqual(planner.plan.call_count, 2)
        print("[v] All components invoked correctly.")

if __name__ == "__main__":
    unittest.main()

