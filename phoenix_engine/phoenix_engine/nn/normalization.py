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
        w_data = Tensor.randn(normalized_shape).data
        b_data = Tensor.randn(normalized_shape).data
        
        # We need a fill operation, but for now we can just use randn and set requires_grad=True
        # In a real engine, we'd use ones() and zeros()
        self.weight = Parameter(Tensor(w_data))
        self.bias = Parameter(Tensor(b_data))
        
    def forward(self, x: Tensor) -> Tensor:
        return x.layernorm(self.weight, self.bias, self.eps)

class Embedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        # Standard normal initialization
        w_data = Tensor.randn(num_embeddings, embedding_dim).data
        self.weight = Parameter(Tensor(w_data))
        
    def forward(self, indices: Tensor) -> Tensor:
        import phoenix_engine._C as pb
        out_data = pb.embedding(self.weight.data, indices.data)
        out = Tensor(out_data, _prev=(self.weight, indices))
        
        def _backward():
            if self.weight.requires_grad:
                # Sparse update would go here
                pass
                
        out._backward = _backward
        return out
