from phoenix.framework.agent.cognition.utils.id import generate_unique_id, generate_timestamped_filename
from helpers import print_step

def test_id_utils():
    print_step("Generating unique ID")
    uid = generate_unique_id()
    assert len(uid) == 32
    
    print_step("Generating timestamped filename")
    fname = generate_timestamped_filename("task")
    assert "task" in fname
    assert len(fname) > 32
    
    print_step("ID utils validation passed")

if __name__ == "__main__":
    test_id_utils()

