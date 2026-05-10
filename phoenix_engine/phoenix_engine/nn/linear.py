from .module import Module, Parameter
from ..tensor import Tensor
import math

class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize weights using randn (simple initialization for now)
        # In a real framework, we'd use Xavier or He initialization
        self.weight = Parameter(Tensor.randn([in_features, out_features]).data)
        
        if bias:
            self.bias = Parameter(Tensor.randn([1, out_features]).data)
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        # If x is > 2D (e.g. [B, T, C]), we need to flatten the batch dimensions for matmul
        original_shape = x.shape
        if len(original_shape) > 2:
            flat_batch = 1
            for d in original_shape[:-1]:
                flat_batch *= d
            x = x.contiguous().view(flat_batch, original_shape[-1])

        out = x.matmul(self.weight)
        
        if len(original_shape) > 2:
            out_shape = list(original_shape[:-1]) + [self.out_features]
            out = out.view(*out_shape)
            
        if self.bias is not None:
            # Broadcast addition
            out = out + self.bias
        return out

    def __repr__(self):
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None})"
