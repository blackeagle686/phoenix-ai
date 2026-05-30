import asyncio
from phoenix.framework.agent.memory.short_term.stm_manager import ShortTermMemoryManager
from helpers import print_step

async def test_stm_manager():
    print_step("Initializing ShortTermMemoryManager with limit 2")
    manager = ShortTermMemoryManager(max_cells=2)
    
    print_step("Adding first message")
    await manager.add("session_1", {"role": "user", "content": "msg 1"})
    
    print_step("Adding second message")
    await manager.add("session_1", {"role": "assistant", "content": "msg 2"})
    
    print_step("Adding third message (triggers eviction)")
    await manager.add("session_1", {"role": "user", "content": "msg 3"})
    
    print_step("Verifying size is 2")
    cells = await manager.get("session_1")
    assert len(cells) == 2
    assert cells[0].content == "msg 2"
    assert cells[1].content == "msg 3"
    
    print_step("Verifying context string")
    context = await manager.get_context_string("session_1")
    assert "Assistant: msg 2" in context
    assert "User: msg 3" in context
    
    print_step("STM Manager validation passed")

if __name__ == "__main__":
    asyncio.run(test_stm_manager())

