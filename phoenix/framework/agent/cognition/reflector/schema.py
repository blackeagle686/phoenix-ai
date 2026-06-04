from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Union
from enum import Enum
from uuid import UUID, uuid4

class Refle

class BaseReflectorMeta(BaseModel):
    rateing: int = Field(..., description="rateing of solution")
    feedback: str = Field(..., description="feedback")
    confidance: float = Field(..., description="confidance of solution")
    reasoning: str = Field(..., description="reasoning of solution")
    
