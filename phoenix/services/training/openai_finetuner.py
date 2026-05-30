import os
from typing import Optional, Dict, Any
from phoenix.services.training.base import BaseFineTuner
from phoenix.core.config import config
from phoenix.services.observability.logger import get_logger
from openai import AsyncOpenAI

logger = get_logger("Phoenix AI.Training")

class OpenAIFineTuner(BaseFineTuner):
    """
    Implements cloud fine-tuning using the OpenAI fine-tuning API.
    """
    
    def __init__(self):
        self.api_key = getattr(config, "OPENAI_API_KEY", None)
        self.client = None

    async def _init_client(self):
        if not self.client:
            if not self.api_key:
                raise RuntimeError("OPENAI_API_KEY is missing. Cannot use OpenAI fine-tuning.")
            self.client = AsyncOpenAI(api_key=self.api_key)

    async def train(
        self, 
        dataset_path: str, 
        model_id: Optional[str] = None, 
        output_dir: Optional[str] = None,
        config_override: Optional[Dict[str, Any]] = None
    ) -> str:
        await self._init_client()
        model_id = model_id or "gpt-4o-mini-2024-07-18"
        
        logger.info(f"Starting OpenAI fine-tuning job with dataset: {dataset_path}")

        try:
            # 1. Upload the training file
            with open(dataset_path, "rb") as f:
                file_info = await self.client.files.create(
                    file=f,
                    purpose="fine-tune"
                )
            
            file_id = file_info.id
            logger.info(f"Uploaded training file. ID: {file_id}")

            # 2. Create the fine-tuning job
            job = await self.client.fine_tuning.jobs.create(
                training_file=file_id,
                model=model_id,
                hyperparameters=config_override.get("hyperparameters", {}) if config_override else None
            )
            
            job_id = job.id
            logger.info(f"OpenAI Fine-tuning job created. ID: {job_id}")
            return job_id

        except Exception as e:
            logger.error(f"OpenAI fine-tuning failed to start: {e}")
            raise e

    async def get_status(self, job_id: str) -> Dict[str, Any]:
        await self._init_client()
        try:
            job = await self.client.fine_tuning.jobs.retrieve(job_id)
            return {
                "status": job.status,
                "model": job.model,
                "fine_tuned_model": job.fine_tuned_model,
                "created_at": job.created_at,
                "finished_at": job.finished_at,
                "error": job.error
            }
        except Exception as e:
            logger.error(f"Could not retrieve OpenAI job {job_id}: {e}")
            return {"status": "error", "error": str(e)}

