from typing import List
from .tensor import Tensor

class SGD:
    def __init__(self, parameters: List[Tensor], lr: float = 0.01):
        self.parameters = parameters
        self.lr = lr
        
    def zero_grad(self):
        for p in self.parameters:
            p.grad = None
            
    def step(self):
        import _phoenix_backend as pb
        for p in self.parameters:
            if p.grad is not None:
                # p = p - lr * p.grad
                # Since we don't have in-place operations yet, we will manually
                # re-assign the underlying C++ TensorData pointer to update the weight.
                # Compute the step
                step = p.grad * self.lr
                # Compute the new data
                new_data = pb.sub(p.data, step.data)
                # Reassign
                p.data = new_data
