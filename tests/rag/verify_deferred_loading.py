import asyncio
import sys
import os

# Add the backend directory to sys.path to allow importing phoenix
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.append(backend_dir)

from phoenix import init_phoenix, startup_phoenix, container

async def test_deferred_loading():
    print("--- Testing Deferred Loading (Defaults) ---")
    # By default, local=False, vlm=False
    init_phoenix()
    await startup_phoenix()
    
    llm_local = container.get("llm_local")
    vlm_local = container.get("vlm_local")
    embeddings = container.get("embeddings")
    
    print(f"LLM Local HF Model loaded: {llm_local.hf_model is not None}")
    print(f"VLM Local HF Model loaded: {vlm_local.hf_model is not None}")
    print(f"Embeddings Model loaded: {embeddings._model is not None}")
    
    assert llm_local.hf_model is None
    assert vlm_local.hf_model is None
    assert embeddings._model is None
    print("[+] Success: Models not loaded by default.")

    print("\n--- Testing Explicit Loading (Local LLM) ---")
    # Note: Actually initializing a real model might be too slow for this test, 
    # but we can check the config flag was set.
    from phoenix.core.config import config
    init_phoenix(local=True)
    print(f"Config LOAD_LOCAL_LLM: {config.LOAD_LOCAL_LLM}")
    assert config.LOAD_LOCAL_LLM is True
    
    print("\n[!] All verification checks passed (logical).")

if __name__ == "__main__":
    asyncio.run(test_deferred_loading())

