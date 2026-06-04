from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Union
from enum import Enum
from uuid import UUID, uuid4

class ReflectorType(str, Enum):
    TASK = "task"
    PROBLEM = "problem"
    SOLUTION = "solution"

class ReflectorInputSchema(BaseModel):
    reflector_type: ReflectorType = Field(..., description="type of reflector")
    description: str = Field(..., description="description of reflector")
    context: Dict[str, Any] = Field(..., description="context of reflector")
    


class BaseReflectorMeta(BaseModel):
    rateing: int = Field(..., description="rateing of solution")
    feedback: str = Field(..., description="feedback")
    confidance: float = Field(..., description="confidance of solution")
    reasoning: str = Field(..., description="reasoning of solution")
    
