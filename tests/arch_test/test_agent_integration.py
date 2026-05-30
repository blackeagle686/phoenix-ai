import asyncio
from phoenix.framework.agent.core.agent import Agent
from phoenix.services.llm.openai import OpenAILLM
from helpers import print_step

async def test_agent_integration_real():
    print_step("Initializing Agent with Custom LongCat LLM")
    
    # Configure LLM with specific credentials for the test
    custom_llm = OpenAILLM(
        api_key="ak_2yp3Xw1Ny7ky2pF7er9x93ZO9jj6G",
        base_url="https://api.longcat.chat/openai",
        model="LongCat-Flash-Lite"
    )
    
    # Initialize the Agent with the custom LLM
    agent = Agent(llm=custom_llm)
    
    # Explicitly initialize the LLM client
    await agent.llm.init()
    
    print_step("Running Agent with prompt: 'say hello'")
    # Using a simple prompt to minimize token usage
    result = await agent.run("say hello", session_id="real_test_session", max_iterations=2)
    
    print_step("Verifying final result")
    print(f"Final Answer: {result}")
    assert len(result) > 0
    
    print_step("Verifying memory persistence in real memory layer")
    context = await agent.memory.get_full_context("real_test_session")
    print(f"Memory Context Sample: {context[:200]}...")
    assert len(context) > 0
    
    print_step("Agent Real Integration Test passed")

if __name__ == "__main__":
    asyncio.run(test_agent_integration_real())

