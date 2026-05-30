import unittest
import asyncio
import os
import shutil
from phoenix.main import init_phoenix, startup_phoenix
from phoenix.services.llm import OpenAILLM
from phoenix.framework.agent.core import Agent
from phoenix.framework.agent.memory.hybrid import HybridMemory

class FullAgentWorkflowTest(unittest.IsolatedAsyncioTestCase):
    """
    End-to-end integration test for the Phoenix AI Agent using real OpenAI LLM.
    Tests the complete cognitive cycle, memory tiers, and tool execution.
    """
    
    async def asyncSetUp(self):
        # Initialize the Phoenix framework
        init_phoenix()
        await startup_phoenix()
        
        # Create a test directory for file tools
        self.test_dir = "./test_agent_workspace"
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        os.makedirs(self.test_dir, exist_ok=True)
        
        # Initialize Real OpenAI LLM and Memory
        self.llm = OpenAILLM()
        self.memory = HybridMemory()
        self.agent = Agent(llm=self.llm, memory=self.memory)

    async def asyncTearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    async def test_full_autonomous_workflow(self):
        print("\n" + "="*60)
        print("PHOENIX AI: REAL OPENAI AGENT WORKFLOW TEST")
        print("="*60)

        # Prompt that requires multiple steps: write then read
        prompt = (
            f"Please create a new file at '{self.test_dir}/workflow_real.txt' "
            "with the content 'Phoenix is alive!'. After writing it, read the file "
            "to verify the content is correct."
        )
        
        print(f"\n[*] Starting Full Workflow with Prompt: '{prompt}'")
        
        # Run the agent in autonomous mode
        result = await self.agent.run(prompt, mode="plan", max_iterations=5)
        
        print("\n[v] Workflow Finished.")
        print(f"[*] Final Agent Report:\n{result}")

        # --- VERIFICATION ---
        
        # 1. Verify filesystem side effects
        file_path = os.path.join(self.test_dir, "workflow_real.txt")
        self.assertTrue(os.path.exists(file_path), "File should have been created by the Agent.")
        
        with open(file_path, "r") as f:
            content = f.read()
            self.assertIn("Phoenix is alive", content)
        print("[v] Filesystem verification passed.")

        # 2. Verify Memory state (Reflection should contain some insight)
        reflections = self.memory.reflection.get_reflections()
        print(f"[*] Agent Reflections:\n{reflections}")
        self.assertTrue(len(reflections) > 0)
        
        # 3. Verify Session state (Objective should be stored)
        obj = self.memory.session.get("current_objective")
        self.assertIsNotNone(obj)
        print("[v] Session and Reflection memory verified.")

        print("\n" + "="*60)
        print("REAL WORKFLOW TEST COMPLETE: SUCCESS")
        print("="*60 + "\n")

if __name__ == "__main__":
    unittest.main()

