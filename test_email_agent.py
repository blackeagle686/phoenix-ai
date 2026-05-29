import asyncio
from phoenix.main import init_phoenix
from phoenix.framework.agent import Agent
from phoenix.framework.agent.core.profile import AgentProfile
from phoenix.framework.agent.tools.bank.productivity.email import EmailTool

async def test():
    # 1. Initialize the SDK (loads .env automatically)
    init_phoenix()
    
    # 2. Create the agent profile
    profile = AgentProfile(
        name="EmailBot",
        role="Outreach Assistant",
        system_prompt=(
            "You are an AI assistant. You have access to the EmailTool. "
            "When requested, use it to send an email to the user's requested address. "
            "If the tool returns an error about missing credentials, simply tell the user what they need to configure."
        ),
        max_iterations=5
    )
    
    # 3. Instantiate the agent with the EmailTool
    agent = Agent(profile=profile, tools=[EmailTool()])
    
    print("[*] Agent is starting...")
    print("[*] Request: Send an email to mathematecs1@gmail.com with content 'hello from your agent'\n")
    
    response = await agent.run(
        "Please send an email to mathematecs1@gmail.com with the subject 'Phoenix Test' and the content 'hello from your agent'.",
        mode="plan" # Force planning mode so it uses the tool instead of just chatting
    )
    
    print("\nFinal Agent Response:")
    print("=====================")
    print(response)

if __name__ == "__main__":
    asyncio.run(test())
