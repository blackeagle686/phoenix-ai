import asyncio
import os
import sys

# Add project root to path
sys.path.insert(0, os.getcwd())

from phoenix.core.container import Container

def test_container_flow():
    print("\n" + "="*50)
    print("PHOENIX AI: CONTAINER SERVICE TEST")
    print("="*50)

    # 1. Initialization
    print("\n[1/3] Initializing Container...")
    c = Container()
    print("[v] Container created.")

    # 2. Registration
    print("\n[2/3] Registering mock services...")
    class MockService:
        def __init__(self, val): self.val = val
    
    c.register("test_service", MockService(42))
    print("[v] Registered 'test_service'.")

    # 3. Retrieval
    print("\n[3/3] Retrieving services...")
    try:
        service = c.get("test_service")
        print(f"[v] Success! Retrieved service with value: {service.val}")
        
        # Test failure
        print("[*] Testing error handling for missing service...")
        try:
            c.get("missing_link")
        except KeyError as e:
            print(f"[v] Caught expected error: {e}")

    except Exception as e:
        print(f"[!] Container Test Error: {e}")

    print("\n" + "="*50)
    print("CONTAINER SERVICE TEST COMPLETE")
    print("="*50 + "\n")

if __name__ == "__main__":
    test_container_flow()

