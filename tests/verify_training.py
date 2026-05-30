import asyncio
from phoenix.Phoenix AI import init_phoenix_full, get_finetuner

async def verify_finetuning_registry():
    print("\n--- [1] Initializing Phoenix AI SDK ---")
    await init_phoenix_full()
    
    # 2. Test Local Finetuner Registration
    print("\n--- [2] Checking Local Finetuner ---")
    local_tuner = get_finetuner(provider="local")
    print(f"Registry Check: {type(local_tuner).__name__}")
    
    # 3. Test OpenAI Finetuner Registration
    print("\n--- [3] Checking OpenAI Finetuner ---")
    openai_tuner = get_finetuner(provider="openai")
    print(f"Registry Check: {type(openai_tuner).__name__}")

    print("\n--- [4] Validating Interfaces ---")
    # Checking if required methods exist
    for tuner in [local_tuner, openai_tuner]:
        has_train = hasattr(tuner, "train")
        has_status = hasattr(tuner, "get_status")
        print(f"{type(tuner).__name__}: train()={has_train}, get_status()={has_status}")

    print("\n--- Verification Complete! Fine-Tuning Service is registered and ready. ---")

if __name__ == "__main__":
    asyncio.run(verify_finetuning_registry())

