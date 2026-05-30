from phoenix.framework.agent.memory.adapter import InteractiveMemoryAdapter
from helpers import print_step

class MockCustomMemory:
    async def add_interaction(self, session_id, role, content, metadata=None):
        return "added"
    async def get_full_context(self, session_id, query=""):
        return "context"

def test_memory_adapter():
    print_step("Initializing InteractiveMemoryAdapter")
    custom = MockCustomMemory()
    adapter = InteractiveMemoryAdapter(custom)
    
    print_step("Verifying session/reflection proxies")
    assert hasattr(adapter, "session")
    assert hasattr(adapter, "reflection")
    
    print_step("Memory Adapter validation passed")

if __name__ == "__main__":
    test_memory_adapter()

