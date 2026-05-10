from typing import List, Tuple, Optional
from .tensor import Tensor
import _phoenix_backend as pb
import math

class SGD:
    def __init__(self, parameters: List[Tensor], lr: float = 0.01):
        self.parameters = parameters
        self.lr = lr
        
    def zero_grad(self):
        for p in self.parameters:
            p.grad = None
            
    def step(self):
        for p in self.parameters:
            if p.grad is not None:
                step = p.grad * self.lr
                p.data = pb.sub(p.data, step.data)

class AdamW:
    def __init__(self, parameters: List[Tensor], lr: float = 0.001, betas: Tuple[float, float] = (0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.01):
        self.parameters = parameters
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        
        self.m = [Tensor.zeros(*p.shape) for p in parameters]
        self.v = [Tensor.zeros(*p.shape) for p in parameters]
        
    def zero_grad(self):
        for p in self.parameters:
            p.grad = None
            
    def step(self):
        self.t += 1
        b1, b2 = self.betas
        
        for i, p in enumerate(self.parameters):
            if p.grad is None:
                continue
            
            # Weight decay: p = p * (1 - lr * wd)
            if self.weight_decay != 0:
                decay_factor = 1.0 - self.lr * self.weight_decay
                p.data = pb.multiply_scalar(p.data, decay_factor)
            
            # m = b1 * m + (1 - b1) * grad
            self.m[i] = self.m[i] * b1 + p.grad * (1.0 - b1)
            # v = b2 * v + (1 - b2) * grad^2
            self.v[i] = self.v[i] * b2 + (p.grad * p.grad) * (1.0 - b2)
            
            # Bias correction
            m_hat = self.m[i] * (1.0 / (1.0 - b1**self.t))
            v_hat = self.v[i] * (1.0 / (1.0 - b2**self.t))
            
            # p = p - lr * m_hat / (sqrt(v_hat) + eps)
            # We don't have sqrt yet! 
            # I need to add sqrt to C++ backend.
            pass
