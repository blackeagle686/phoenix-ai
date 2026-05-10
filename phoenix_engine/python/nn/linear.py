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
        out = x.matmul(self.weight)
        if self.bias is not None:
            out = out + self.bias
        return out

    def __repr__(self):
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None})"




