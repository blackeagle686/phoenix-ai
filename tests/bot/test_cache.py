import asyncio
import os
import sys

# Add project root to path
sys.path.insert(0, os.getcwd())

from phoenix.services.cache.redis_cache import RedisCache

async def test_cache_flow():
    print("\n" + "="*50)
    print("PHOENIX AI: CACHE SERVICE TEST")
    print("="*50)

    # 1. Initialization
    print("\n[1/3] Initializing Redis Cache...")
    cache = RedisCache()
    
    try:
        await cache.init()
        print("[v] Connection to Redis established (or initialized).")

        # 2. Set Value
        print("\n[2/3] Setting value in cache...")
        key = "test_key_001"
        value = {"status": "success", "code": 200, "message": "Cached response"}
        await cache.set(key, value, ttl=60)
        print(f"[v] Success! Key '{key}' set.")

        # 3. Get Value
        print("\n[3/3] Retrieving value from cache...")
        cached_value = await cache.get(key)
        
        if cached_value:
            print(f"[v] Success! Retrieved: {cached_value}")
        else:
            print("[!] Error: Value not found in cache.")

        # Cleanup
        print("\n[*] Deleting test key...")
        await cache.delete(key)
        print("[v] Done.")

    except Exception as e:
        print(f"[!] Cache Service Error: {e}")
        print("[*] Note: This test requires a running Redis instance or valid REDIS_URL.")

    print("\n" + "="*50)
    print("CACHE SERVICE TEST COMPLETE")
    print("="*50 + "\n")

if __name__ == "__main__":
    asyncio.run(test_cache_flow())

