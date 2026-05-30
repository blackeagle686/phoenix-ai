import asyncio
import os
import sys

# Add project root to path
sys.path.insert(0, os.getcwd())

from phoenix.core.lifecycle import LifecycleManager

async def test_lifecycle_flow():
    print("\n" + "="*50)
    print("PHOENIX AI: LIFECYCLE HOOKS TEST")
    print("="*50)

    # 1. Initialization
    print("\n[1/4] Initializing Lifecycle Manager...")
    lm = LifecycleManager()
    
    # State for verification
    executed = []

    # 2. Registering Hooks
    print("\n[2/4] Registering hooks...")
    
    async def startup_1():
        print("[*] Startup Hook 1 running...")
        executed.append("start1")
        
    async def startup_2():
        print("[*] Startup Hook 2 running...")
        executed.append("start2")
        
    async def failing_hook():
        print("[!] Failing Hook running (should not stop others)...")
        raise RuntimeError("Controlled failure")

    lm.on_startup(startup_1)
    lm.on_startup(failing_hook)
    lm.on_startup(startup_2)
    
    # 3. Executing Startup (Sequential)
    print("\n[3/4] Running Startup Hooks (Sequential)...")
    await lm.startup(parallel=False)
    
    if "start1" in executed and "start2" in executed:
        print("[v] Success! Both healthy startup hooks executed despite failure in middle.")
    else:
        print("[!] Error: Hooks were skipped.")

    # 4. Executing Parallel
    executed.clear()
    print("\n[4/4] Running Startup Hooks (Parallel)...")
    await lm.startup(parallel=True)
    
    if "start1" in executed and "start2" in executed:
        print("[v] Success! Parallel hooks completed.")
    else:
        print("[!] Error: Parallel execution failed.")

    print("\n" + "="*50)
    print("LIFECYCLE HOOKS TEST COMPLETE")
    print("="*50 + "\n")

if __name__ == "__main__":
    asyncio.run(test_lifecycle_flow())

