import asyncio
from phoenix.Phoenix AI import init_phoenix, get_vlm_pipeline, get_insight_engine
from phoenix.services.observability.logger import get_logger

logger = get_logger("Verification.Observability")

async def verify_observability():
    # 1. Initialize SDK
    print("\n--- [1] Initializing Phoenix AI SDK ---")
    services = await init_phoenix()
    
    # 2. Test VLM Pipeline with Logging
    print("\n--- [2] Testing VLM Pipeline (Local/Mock) ---")
    vlm = get_vlm_pipeline()
    # Using a fake path for demonstration - it will trigger error tracing if it doesn't exist, 
    # but we'll use a string check for mock mode in some cases.
    # For now, let's just trigger a generate call.
    try:
        # We'll use a small timeout or mock check
        response = await vlm.ask("What is in this image?", "/content/cloud.pdf", use_rag=False)
        print(f"VLM Response: {response[:100]}...")
    except Exception as e:
        print(f"Expected failure handled: {e}")

    # 3. Test Insight Engine with Cache & Retrieval Logging
    print("\n--- [3] Testing Insight Engine ---")
    engine = get_insight_engine()
    response = await engine.query("Tell me about the cloud features.")
    print(f"Insight Response: {response[:100]}...")
    
    # Second query to trigger Cache Hit logging
    print("\n--- [4] Testing Cache Hit Logging ---")
    await engine.query("Tell me about the cloud features.")

    print("\n--- Verification Complete. Check terminal output for 'META:' logs! ---")

if __name__ == "__main__":
    asyncio.run(verify_observability())

