import asyncio
from phoenix.framework.agent.memory.hybrid.hybrid_manager import HybridMemoryManager
from helpers import print_step, MockLLM

async def test_hybrid_manager():
    print_step("Initializing HybridMemoryManager")
    manager = HybridMemoryManager()
    
    print_step("Adding interaction")
    await manager.add_interaction("session_1", "user", "How are you?")
    
    print_step("Retrieving full context")
    context = await manager.get_full_context("session_1")
    
    print_step("Verifying context contains STM data")
    assert "User: How are you?" in context
    
    print_step("Hybrid Manager validation passed")

if __name__ == "__main__":
    asyncio.run(test_hybrid_manager())

