import asyncio
import os
import uuid
from phoenix.framework.agent.core.agent import Agent
from phoenix.services.llm.openai import OpenAILLM
from phoenix.core.config import config

async def test_agent_full_flow():
    print("\\n--- Testing Agent Full Flow ---")
    
    session_id = f"test-session-{uuid.uuid4()}"
    test_file = "tests/test_021/agent_test_output.txt"
    
    if os.path.exists(test_file):
        os.remove(test_file)

    try:
        # 1. Initialize Agent with specific LongCat config
        config.OPENAI_API_KEY = "ak_2yp3Xw1Ny7ky2pF7er9x93ZO9jj6G"
        config.OPENAI_BASE_URL = "https://api.longcat.chat/openai"
        config.OPENAI_LLM_MODEL = "LongCat-Flash-Chat"
        
        agent = Agent()

        # 2. Run a task that requires tools (Writing a file)
        task = f"Please create a text file at '{test_file}' containing the string 'Phoenix Agent Integration Test'. Then read it back to verify."
        print(f"Task: {task}")
        
        result = await agent.run(task, session_id=session_id)
        print(f"Agent Final Answer: {result}")

        # 3. Verify tool execution
        assert os.path.exists(test_file), "Agent failed to create the file using tools."
        with open(test_file, "r") as f:
            content = f.read()
            assert "Phoenix Agent Integration Test" in content, f"File content mismatch: {content}"
        
        print("File verification successful.")

        # 4. Verify Memory Integration
        memory_check_task = "What was the path of the file you just created?"
        print(f"Task: {memory_check_task}")
        
        result2 = await agent.run(memory_check_task, session_id=session_id)
        print(f"Agent Final Answer: {result2}")
        assert "agent_test_output.txt" in result2, "Agent failed to remember the file path from its previous execution."

        print("\\n✅ Agent Full Flow Test Passed!")

    finally:
        # Cleanup
        if os.path.exists(test_file):
            os.remove(test_file)

if __name__ == "__main__":
    if not config.OPENAI_API_KEY and not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found. Please set it in your environment for full integration testing.")
    else:
        asyncio.run(test_agent_full_flow())

