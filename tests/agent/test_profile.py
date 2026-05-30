import asyncio
import os
import sys

# Add the project root to sys.path so we can import 'phoenix'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from phoenix.framework.agent.core.agent import Agent
from phoenix.framework.agent.core.profile import AgentProfile

async def main():
    print("--- Testing AgentProfile Integration ---")

    # 1. Define a mock profile via a dictionary
    profile_dict = {
        "identity": {
            "name": "TestAdmin-X",
            "id": "test-id-001"
        },
        "role": {
            "title": "System Tester",
            "mission": "To verify that the agent profile framework works as expected."
        },
        "personality": {
            "communication_tone": "robotic, precise",
            "response_style": "minimalist"
        },
        "rules": [
            "Never lie.",
            "Always follow test protocols."
        ],
        "capabilities": [
            "Code analysis",
            "Test generation"
        ],
        "tool_access": [
            "bash_shell_executor"
        ]
    }

    print("\n1. Initializing Agent with Profile Dict...")
    agent = Agent(profile=profile_dict)

    print(f"\n2. Verifying Profile was Loaded:")
    print(f"Agent Profile Name: {agent.profile.identity.name}")
    print(f"Agent Profile Mission: {agent.profile.role.mission}")

    print("\n3. Verifying Injection into Cognition Modules:")
    print(f"Thinker has profile: {agent.thinker.profile is not None}")
    print(f"Planner has profile: {agent.planner.profile is not None}")
    print(f"Analyzer has profile: {agent.analyzer.profile is not None}")
    print(f"Reflector has profile: {agent.reflector.profile is not None}")

    assert agent.thinker.profile.identity.name == "TestAdmin-X", "Thinker profile mismatch!"

    print("\n4. Verifying Prompt Generation String:")
    prompt_str = agent.profile.to_prompt_string()
    print("--------------------------------------------------")
    print(prompt_str)
    print("--------------------------------------------------")

    print("\n[SUCCESS] AgentProfile framework updates are working correctly!")

if __name__ == "__main__":
    asyncio.run(main())

