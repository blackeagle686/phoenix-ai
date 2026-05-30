import asyncio
import os
import sys

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from phoenix.framework.multi_agent.manager import MultiAgentManager
from phoenix.framework.multi_agent.config import MultiAgentConfig, AgentConfig
from phoenix.services.llm.openai import OpenAILLM

async def main():
    print("--- Testing Full Power Multi-Agent Framework ---")
    print("Initializing 3 specialized Hashira agents with Thinker, Planner, Actor, Reflector, and HybridMemory...\n")

    # 1. Define configurations for a powerful 3-Agent Team
    # All agents will use the OpenAILLM backend by default when llm_type="openai"
    
    architect_config = AgentConfig(
        name="Tengen_Architect",
        llm_type="openai",
        profile={
            "identity": {"name": "Tengen", "id": "hashira-arch-01"},
            "role": {"title": "System Architect", "mission": "To design robust and flashy software architectures."},
            "personality": {"communication_tone": "flamboyant and direct", "response_style": "detailed"},
            "rules": ["Always outline a clear step-by-step plan.", "Focus on scalability."],
            "capabilities": ["System Design", "Architecture"],
            "tool_access": ["web_search"]
        }
    )

    coder_config = AgentConfig(
        name="Giyu_Coder",
        llm_type="openai", 
        profile={
            "identity": {"name": "Giyu", "id": "hashira-code-02"},
            "role": {"title": "Lead Developer", "mission": "To write precise, bug-free Python code based on architectural plans."},
            "personality": {"communication_tone": "calm and quiet", "response_style": "code-heavy"},
            "rules": ["Follow PEP 8.", "Do not write bloated code.", "Always include type hints."],
            "capabilities": ["Python", "Algorithms", "Implementation"],
            "tool_access": ["python_repl", "file_write"]
        }
    )

    reviewer_config = AgentConfig(
        name="Shinobu_Reviewer",
        llm_type="openai",
        profile={
            "identity": {"name": "Shinobu", "id": "hashira-rev-03"},
            "role": {"title": "Security & Code Reviewer", "mission": "To find bugs, security flaws, and test the provided code."},
            "personality": {"communication_tone": "cheerful but sharp", "response_style": "analytical"},
            "rules": ["Be thorough in code reviews.", "Highlight any potential edge cases."],
            "capabilities": ["Security audit", "Code review", "Testing"],
            "tool_access": ["python_repl"]
        }
    )

    team_config = MultiAgentConfig(
        team_name="Hashira-OS Task Force",
        agents=[architect_config, coder_config, reviewer_config]
    )

    # 2. Initialize the Team
    print("1. Initializing MultiAgentManager with full cognitive loops...")
    manager = MultiAgentManager(team_config)

    # 3. Verify Team Overview and Capabilities
    print("\n2. Team Roster:")
    for member in manager.get_team_overview():
        print(f"- {member['name']} ({member['role']}): {member['mission']}")

    # 4. Run a Full Collaborative Pipeline to fix a single complex problem
    # Problem: Design, implement, and review a thread-safe LRU Cache.
    print("\n3. Launching Collaborative Pipeline (Architect -> Coder -> Reviewer)...")
    
    problem_statement = (
        "We need a Python implementation of a Thread-Safe LRU (Least Recently Used) Cache. "
        "It needs to have `get`, `put`, and `get_cache_size` methods. "
        "Please provide the full code and a security/performance review."
    )
    
    print(f"\n[Problem]: {problem_statement}\n")

    try:
        # The run_pipeline method will pass the output of one agent as context to the next.
        # Since we are using "auto" or "plan" mode by default in the agent, they will 
        # utilize their full Thinker -> Planner -> Actor -> Reflector loop!
        final_result = await manager.run_pipeline(
            prompt=problem_statement, 
            agent_sequence=["Tengen_Architect", "Giyu_Coder", "Shinobu_Reviewer"]
        )
        
        print("\n==================================================")
        print("FINAL PIPELINE RESULT (From Shinobu_Reviewer):")
        print("==================================================")
        print(final_result)
        
    except Exception as e:
        print(f"\nPipeline failed (ensure your OPENAI_API_KEY is set in .env): {e}")

    print("\n[SUCCESS] Full-power Multi-Agent execution completed!")

if __name__ == "__main__":
    asyncio.run(main())

