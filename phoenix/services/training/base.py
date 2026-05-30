from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

class BaseFineTuner(ABC):
    """
    Base interface for Fine-Tuning services.
    Supports both local (PEFT/trl) and cloud-based (OpenAI) model adaptation.
    """
    
    @abstractmethod
    async def train(
        self, 
        dataset_path: str, 
        model_id: Optional[str] = None, 
        output_dir: Optional[str] = None,
        config_override: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Starts a fine-tuning job.
        Returns a job ID or local path to the fine-tuned model.
        """
        pass

    @abstractmethod
    async def get_status(self, job_id: str) -> Dict[str, Any]:
        """
        Retrieves the status of a specific training job.
        """
        pass

