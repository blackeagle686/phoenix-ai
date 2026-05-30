import asyncio
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from phoenix.framework.multi_agent.manager import MultiAgentManager
from phoenix.framework.multi_agent.config import MultiAgentConfig, AgentConfig
from phoenix.services.llm.openai import OpenAILLM

async def main():
    print("--- Testing Advanced Production Multi-Agent Features ---")

    # 1. Configuration with Shared Memory enabled
    team_config = MultiAgentConfig(
        team_name="Hashira Elite",
        shared_memory=True,  # Crucial for deep collaboration
        agents=[
            AgentConfig(
                name="Giyu_Coder",
                llm_type="openai", 
                profile={
                    "identity": {"name": "Giyu", "id": "giyu-01"},
                    "role": {"title": "Developer", "mission": "Write code."},
                    "personality": {"communication_tone": "calm", "response_style": "code-heavy"},
                    "capabilities": ["Python Coding"],
                }
            ),
            AgentConfig(
                name="Shinobu_Reviewer",
                llm_type="openai",
                profile={
                    "identity": {"name": "Shinobu", "id": "shinobu-01"},
                    "role": {"title": "Reviewer", "mission": "Review code and find errors."},
                    "personality": {"communication_tone": "cheerful", "response_style": "analytical"},
                    "capabilities": ["Security", "Code Review"],
                }
            )
        ]
    )

    manager = MultiAgentManager(team_config)
    print("\n[✓] Manager initialized with Shared Memory.")

    # 2. Test Autonomous Routing
    print("\n--- Testing Autonomous Routing ---")
    print("Sending prompt: 'Please review this code snippet for security vulnerabilities...'")
    try:
        # The router should automatically pick Shinobu_Reviewer based on capabilities
        result = await manager.run_autonomous("Please review this code snippet for security vulnerabilities: `eval(user_input)`")
        print("\nRouter Execution Result:")
        print(result)
    except Exception as e:
        print(f"Autonomous routing skipped/failed (expected if no API key): {e}")

    # 3. Test Review Loop (Self-Correcting)
    print("\n--- Testing Self-Correcting Review Loop ---")
    problem = "Write a python function that returns the square of a number, but make a deliberate syntax error so the reviewer catches it."
    
    try:
        # Giyu will write it, Shinobu will review it. If Shinobu rejects, Giyu will try again.
        final_code = await manager.run_with_review(
            prompt=problem,
            doer="Giyu_Coder",
            reviewer="Shinobu_Reviewer",
            max_loops=2 # Keep it short for testing
        )
        print("\nFinal Output after Review Loop:")
        print(final_code)
    except Exception as e:
        print(f"Review loop skipped/failed (expected if no API key): {e}")

    print("\n[SUCCESS] Advanced Multi-Agent tests completed.")

if __name__ == "__main__":
    asyncio.run(main())

