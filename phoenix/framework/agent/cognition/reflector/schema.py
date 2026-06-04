from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Union, Any
from enum import Enum
from uuid import UUID, uuid4

class ReflectorType(str, Enum):
    TASK = "task"
    PROBLEM = "problem"
    SOLUTION = "solution"

class ReflectorInputSchema(BaseModel):
    reflector_type: ReflectorType = Field(..., description="The type of the item being evaluated (Task, Problem, Solution)")
    target_id: str = Field(..., description="The unique identifier of the target item being evaluated")
    target_content: Any = Field(..., description="The serialized content of the item being evaluated")
    context: Optional[str] = Field(None, description="Overarching objective or context to guide the evaluation")

class BaseReflectorMeta(BaseModel):
    rating: int = Field(..., ge=1, le=10, description="Evaluation rating on a scale of 1 to 10")
    feedback: str = Field(..., description="Constructive feedback or critique")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score of this evaluation between 0.0 and 1.0")
    reasoning: str = Field(..., description="Detailed reasoning for the assigned rating and feedback")

class ReflectorOutputSchema(BaseReflectorMeta):
    """
    Final output returned by the Reflector.
    Inherits rating, feedback, confidence, and reasoning from BaseReflectorMeta.
    """
    pass