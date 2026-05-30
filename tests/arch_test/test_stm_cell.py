from phoenix.framework.agent.memory.short_term.stm_cell import ShortMemoryCell
from helpers import print_step

def test_short_memory_cell():
    print_step("Initializing ShortMemoryCell")
    cell = ShortMemoryCell(
        memory_id="test_id",
        session_id="session_123",
        content="Hello world",
        role="user",
        step=1
    )
    
    print_step("Verifying cell attributes")
    assert cell.memory_id == "test_id"
    assert cell.content == "Hello world"
    assert cell.role == "user"
    assert cell.relevance_score == 1.0
    
    print_step("Cell validation passed")

if __name__ == "__main__":
    test_short_memory_cell()

