import asyncio
from phoenix.framework.agent.cognition.planner.planner import Planner
from helpers import print_step, MockLLM

async def test_planner():
    print_step("Initializing Planner with MockLLM and MockTools")
    
    class MockTools:
        def get_all_tools_info(self): return {}

    llm = MockLLM()
    planner = Planner(llm, MockTools())
    
    print_step("Generating plan for objective")
    # We need to mock parse_llm_json or the llm response to be valid JSON
    llm.generate = lambda p, **k: asyncio.sleep(0, result='{"actions": [{"tool": "test", "kwargs": {}}]}')
    
    plan = await planner.plan("Test objective")
    
    print_step("Verifying plan structure")
    assert "actions" in plan
    assert plan["actions"][0]["tool"] == "test"
    
    print_step("Planner validation passed")

if __name__ == "__main__":
    asyncio.run(test_planner())

