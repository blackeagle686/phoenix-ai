from phoenix.framework.agent.memory.long_term.ltm_cell import LongMemoryCell
from helpers import print_step

def test_long_memory_cell():
    print_step("Initializing LongMemoryCell")
    cell = LongMemoryCell(
        memory_id="ltm_1",
        content="Persistent fact",
        embedding=[0.1, 0.2],
        memory_type="knowledge"
    )
    
    print_step("Verifying attributes")
    assert cell.content == "Persistent fact"
    assert cell.memory_type == "knowledge"
    
    print_step("LongMemoryCell validation passed")

if __name__ == "__main__":
    test_long_memory_cell()

