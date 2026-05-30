import asyncio
from phoenix.framework.agent.cognition.analyzer.analyzer import Analyzer
from helpers import print_step, MockLLM

async def test_analyzer():
    print_step("Initializing Analyzer")
    llm = MockLLM()
    analyzer = Analyzer(llm)
    
    print_step("Mocking LLM response for analyzer")
    llm.generate = lambda p, **k: asyncio.sleep(0, result='{"relevant_files": ["f1.py"], "tech_stack": "Python", "summary": "ok"}')
    
    print_step("Running workspace analysis")
    analysis = await analyzer.analyze_workspace("Test prompt")
    
    print_step("Verifying analysis results")
    assert analysis["tech_stack"] == "Python"
    assert "f1.py" in analysis["relevant_files"]
    
    print_step("Analyzer validation passed")

if __name__ == "__main__":
    asyncio.run(test_analyzer())

