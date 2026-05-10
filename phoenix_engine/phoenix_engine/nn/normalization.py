from typing import Optional
from ..tensor import Tensor
from .module import Module, Parameter
import math

class LayerNorm(Module):
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        
        # Initialize weight to 1 and bias to 0
        self.weight = Parameter(Tensor.ones(normalized_shape))
        self.bias = Parameter(Tensor.zeros(normalized_shape))
        
    def forward(self, x: Tensor) -> Tensor:
        return x.layernorm(self.weight, self.bias, self.eps)
