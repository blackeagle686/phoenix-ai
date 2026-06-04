from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Union
from enum import Enum
from uuid import UUID, uuid4

class BaseReflectorMeta(BaseModel):
    rateing: int = Field(..., description="rateing of solution")
    feedback: str = Field(..., description="feedback")
    
    