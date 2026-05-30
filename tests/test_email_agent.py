import asyncio
from phoenix.main import init_phoenix
from phoenix.framework.agent.tools.registry import ToolRegistry
from phoenix.framework.agent import Agent
from phoenix.framework.agent.core.profile import AgentProfile, Identity, Role, Personality
from phoenix.framework.agent.tools.bank.productivity.email import EmailTool

from phoenix.services.llm.openai import OpenAILLM

import os
os.environ["SMTP_EMAIL"] = "mathematecs1@gmail.com"
os.environ["SMTP_PASSWORD"] = ""

async def test():
    # 1. Initialize the SDK (loads .env automatically)
    init_phoenix()
    
    # 2. Create the agent profile
    profile = AgentProfile(
        identity=Identity(name="EmailBot", id="email-bot-1"),
        role=Role(
            title="Outreach Assistant",
            mission="Send emails to users when requested."
        ),
        personality=Personality(
            communication_tone="Professional",
            response_style="Direct"
        ),
        rules=[
            "If the email tool returns an error about missing credentials, tell the user exactly what is missing."
        ],
        tool_access=["email"]
    )
    
    from phoenix.framework.agent.memory.adapter import InteractiveMemoryAdapter

    registry = ToolRegistry()
    registry.register(EmailTool())
    
    # Initialize the real LLM with Gemini
    llm = OpenAILLM(
        api_key=os.environ.get("GEMINI_API_KEY", ""),
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        model="gemini-2.5-flash"
    )
    await llm.init()
    registry.register(EmailTool())
    
    # 3. Instantiate the agent with the MockLLM and basic memory to avoid VectorDB dependencies
    class BasicMemory:
        async def add_interaction(self, session_id, role, content, metadata=None): pass
        async def get_full_context(self, session_id, query=None): return ""
        
    agent = Agent(llm=llm, memory=BasicMemory(), profile=profile, tools=registry)
    
    print("[*] Agent is starting...")
    print("[*] Request: Send an email to mathematecs123@gmail.com with content 'hello from your agent'\\n")
    
    response = await agent.run(
        "Please send an email to mathematecs123@gmail.com that say hello from our agent *_* add some emogies and so on  ",
        mode="plan" # Force planning mode so it uses the tool instead of just chatting
    )
    
    print("\\nFinal Agent Response:")
    print("=====================")
    print(response)

if __name__ == "__main__":
    asyncio.run(test())

