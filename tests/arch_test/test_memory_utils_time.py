from phoenix.memory.utils.time import current_timestamp, calculate_expiry, has_expired
from helpers import print_step
import time

def test_time_utils():
    print_step("Getting current timestamp")
    now = current_timestamp()
    assert now > 0
    
    print_step("Calculating expiry for 1 second")
    expiry = calculate_expiry(1)
    
    print_step("Verifying not yet expired")
    assert has_expired(expiry) == False
    
    print_step("Waiting for expiry...")
    time.sleep(1.1)
    
    print_step("Verifying now expired")
    assert has_expired(expiry) == True
    
    print_step("Time utils validation passed")

if __name__ == "__main__":
    test_time_utils()

