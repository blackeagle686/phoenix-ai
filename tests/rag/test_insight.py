import asyncio
import os
import sys

# Add project root to path
sys.path.insert(0, os.getcwd())

from phoenix.main import init_phoenix, startup_phoenix, get_vlm_pipeline, get_insight_engine

async def test_insight_flow():
    print("\n" + "="*50)
    print("PHOENIX AI: INSIGHT & VLM SERVICE TEST")
    print("="*50)

    # 1. Initialization
    print("\n[1/3] Initializing Insight Services...")
    init_phoenix(local=True)
    await startup_phoenix()
    
    # 2. Test VLM Pipeline
    print("\n[2/3] Testing VLM Pipeline...")
    vlm = get_vlm_pipeline()
    
    # Create a dummy image file for testing
    dummy_image = "tests/test_image.png"
    with open(dummy_image, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82")
    
    print(f"[*] Created {dummy_image}")
    prompt = "Describe what is in this image."
    print(f"[*] Sending prompt to VLM: {prompt}")
    
    try:
        # Will return mock response in local mode
        response = await vlm.ask(prompt, dummy_image)
        print(f"[v] Success! VLM Response: {response}")
    except Exception as e:
        print(f"[!] VLM Error: {e}")

    # 3. Test Insight Engine
    print("\n[3/3] Testing Insight Engine...")
    insight = get_insight_engine()
    query = "Analyze the codebase for security vulnerabilities."
    print(f"[*] Sending query to Insight Engine: {query}")
    
    try:
        response = await insight.query(query)
        print(f"[v] Success! Insight Response: {response}")
    except Exception as e:
        print(f"[!] Insight Engine Error: {e}")

    # Cleanup
    if os.path.exists(dummy_image):
        os.remove(dummy_image)

    print("\n" + "="*50)
    print("INSIGHT & VLM SERVICE TEST COMPLETE")
    print("="*50 + "\n")

if __name__ == "__main__":
    asyncio.run(test_insight_flow())

