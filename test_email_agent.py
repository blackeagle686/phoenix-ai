import asyncio
from phoenix.main import init_phoenix
from phoenix.framework.agent.tools.registry import ToolRegistry
from phoenix.framework.agent import Agent
from phoenix.framework.agent.core.profile import AgentProfile, Identity, Role, Personality
from phoenix.framework.agent.tools.bank.productivity.email import EmailTool

class MockLLM:
    def __init__(self):
        self.call_count = 0
        
    async def generate(self, prompt, **kwargs):
        self.call_count += 1
        if self.call_count == 1:
            return """
            ```json
            {
                "actions": [
                    {
                        "tool": "email",
                        "kwargs": {
                            "recipient": "mathematecs1@gmail.com",
                            "subject": "Phoenix Test",
                            "body": "hello from your agent"
                        }
                    }
                ]
            }
            ```
            """
        else:
            return '{"actions": [{"tool": "finish"}]}'

    async def generate_stream(self, prompt, **kwargs):
        yield "Thinking..."

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
    
    registry = ToolRegistry()
    registry.register(EmailTool())
    
    # 3. Instantiate the agent with the MockLLM
    agent = Agent(llm=MockLLM(), profile=profile, tools=registry)
    
    print("[*] Agent is starting...")
    print("[*] Request: Send an email to mathematecs1@gmail.com with content 'hello from your agent'\\n")
    
    response = await agent.run(
        "Please send an email to mathematecs1@gmail.com with the subject 'Phoenix Test' and the content 'hello from your agent'.",
        mode="plan" # Force planning mode so it uses the tool instead of just chatting
    )
    
    print("\\nFinal Agent Response:")
    print("=====================")
    print(response)

if __name__ == "__main__":
    asyncio.run(test())


