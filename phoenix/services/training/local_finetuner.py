import os
try:
    import torch
except ImportError:
    torch = None
from typing import Optional, Dict, Any
from phoenix.services.training.base import BaseFineTuner
from phoenix.core.config import config
from phoenix.services.observability.logger import get_logger

logger = get_logger("Phoenix AI.Training")

class LocalFineTuner(BaseFineTuner):
    """
    Implements local fine-tuning using PEFT (LoRA/QLoRA) and Hugging Face TRL.
    """
    
    def __init__(self):
        self.jobs = {} # In-memory job status tracking for current session

    async def train(
        self, 
        dataset_path: str, 
        model_id: Optional[str] = None, 
        output_dir: Optional[str] = None,
        config_override: Optional[Dict[str, Any]] = None
    ) -> str:
        model_id = model_id or config.LOCAL_LLM_MODEL
        output_dir = output_dir or "./finetuned_models"
        
        job_id = f"local_{os.urandom(4).hex()}"
        self.jobs[job_id] = {"status": "starting", "model": model_id}
        
        logger.info(f"Starting local training job: {job_id} for model {model_id}")

        # This runs synchronously because training is compute-heavy. 
        # In a real production SDK, we would offload this to a separate process or background worker.
        try:
            from transformers import (
                AutoModelForCausalLM, 
                AutoTokenizer, 
                BitsAndBytesConfig, 
                TrainingArguments
            )
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            from trl import SFTTrainer
            from datasets import load_dataset

            if torch is None:
                raise ImportError("torch is required for local training. Install with: pip install phx-ashborn[local]")

            # 1. Hardware Detection
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[*] Training on device: {device}")

            # 2. BitsAndBytes Config (QLoRA)
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            ) if device == "cuda" else None

            # 3. Load Model & Tokenizer
            print(f"[*] Loading base model: {model_id}...")
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map="auto" if device == "cuda" else None,
                trust_remote_code=True
            )
            model.config.use_cache = False
            
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            tokenizer.pad_token = tokenizer.eos_token

            # 4. PEFT Config (LoRA)
            peft_config = LoraConfig(
                r=8,
                lora_alpha=32,
                target_modules=["q_proj", "v_proj"], # Common targets for Llama/Qwen
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM"
            )

            if device == "cuda":
                model = prepare_model_for_kbit_training(model)
            model = get_peft_model(model, peft_config)

            # 5. Load Dataset
            print(f"[*] Loading dataset from: {dataset_path}...")
            ext = dataset_path.split('.')[-1]
            if ext == "jsonl":
                dataset = load_dataset("json", data_files=dataset_path, split="train")
            elif ext == "csv":
                dataset = load_dataset("csv", data_files=dataset_path, split="train")
            else:
                # Assume a directory of text files for simplest SFT
                dataset = load_dataset("text", data_dir=dataset_path, split="train")

            # 6. Training Arguments
            training_args = TrainingArguments(
                output_dir=os.path.join(output_dir, job_id),
                per_device_train_batch_size=1,
                gradient_accumulation_steps=4,
                learning_rate=2e-4,
                num_train_epochs=1,
                logging_steps=10,
                save_steps=100,
                fp16=True if device == "cuda" else False,
                report_to="none"
            )

            # 7. Start Training
            print(f"[*] Launching SFTTrainer for {job_id}...")
            self.jobs[job_id]["status"] = "running"
            
            trainer = SFTTrainer(
                model=model,
                train_dataset=dataset,
                peft_config=peft_config,
                dataset_text_field="text",
                max_seq_length=512,
                tokenizer=tokenizer,
                args=training_args,
            )

            trainer.train()
            
            # 8. Save
            final_path = os.path.join(output_dir, job_id, "final_lora")
            trainer.model.save_pretrained(final_path)
            self.jobs[job_id]["status"] = "completed"
            self.jobs[job_id]["path"] = final_path
            
            logger.info(f"Training completed successfully. Model saved to: {final_path}")
            return job_id

        except Exception as e:
            self.jobs[job_id]["status"] = "failed"
            self.jobs[job_id]["error"] = str(e)
            logger.error(f"Training job {job_id} failed: {e}")
            raise e

    async def get_status(self, job_id: str) -> Dict[str, Any]:
        return self.jobs.get(job_id, {"status": "not_found"})
