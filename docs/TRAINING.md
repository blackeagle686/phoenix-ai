# Fine-Tuning Service Guide

The **Fine-Tuning Service** allows you to adapt LLMs and VLMs to your specific domain using local hardware or cloud APIs.

## 1. Local Fine-Tuning (LoRA / QLoRA)
Efficiently train models on your own GPU using Parameter-Efficient Fine-Tuning (PEFT).

### Prerequisites
- `pip install peft trl bitsandbytes accelerate`
- A GPU with >= 8GB VRAM (for small models) or >= 16GB (for 7B models).

### Usage
```python
from phoenix import init_phoenix_full, get_finetuner

async def train_locally():
    await init_phoenix_full()
    finetuner = get_finetuner(provider="local")

    # Start training with a JSONL dataset
    # Dataset format: {"text": "Instruction: ... Context: ... Response: ..."}
    job_id = await finetuner.train(
        dataset_path="./data/my_dataset.jsonl",
        model_id="Qwen/Qwen2-1.5B-Instruct",
        output_dir="./training_outputs"
    )
    
    print(f"Training started: {job_id}")
```

---

## 2. Cloud Fine-Tuning (OpenAI)
Adapt models in the cloud without needing local GPU resources.

### Usage
```python
from phoenix import init_phoenix_full, get_finetuner

async def train_cloud():
    await init_phoenix_full()
    finetuner = get_finetuner(provider="openai")

    # Starts a cloud job
    job_id = await finetuner.train(
        dataset_path="./data/cloud_dataset.jsonl",
        model_id="gpt-4o-mini-2024-07-18"
    )
    
    # Monitor status
    status = await finetuner.get_status(job_id)
    print(f"Job Status: {status['status']}")
```

---

## 3. Dataset Format
- **LLM (Text):** JSONL files where each line is a dictionary with a `"text"` key containing the full prompt and response.
- **VLM (Vision):** Support for image-text datasets follows standard Hugging Face formats.

## 4. Configuration
You can customize the following in your `.env` file:
```env
TRAINING_BATCH_SIZE=1
TRAINING_EPOCHS=3
TRAINING_LEARNING_RATE=2e-4
TRAINING_LORA_R=8
```

