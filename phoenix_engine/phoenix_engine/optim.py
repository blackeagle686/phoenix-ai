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
            
            # 1. Weight decay (AdamW style: decay applied before moment updates)
            if self.weight_decay != 0:
                # p = p - lr * weight_decay * p
                decay_step = p.data.multiply_scalar(self.lr * self.weight_decay)
                p.data = pb.sub(p.data, decay_step)
            
            # 2. Update biased first moment estimate
            # m = b1 * m + (1 - b1) * grad
            m_term1 = self.m[i].data.multiply_scalar(b1)
            m_term2 = p.grad.data.multiply_scalar(1.0 - b1)
            self.m[i].data = pb.add(m_term1, m_term2)
            
            # 3. Update biased second raw moment estimate
            # v = b2 * v + (1 - b2) * grad^2
            v_term1 = self.v[i].data.multiply_scalar(b2)
            grad_sq = pb.multiply(p.grad.data, p.grad.data)
            v_term2 = grad_sq.multiply_scalar(1.0 - b2)
            self.v[i].data = pb.add(v_term1, v_term2)
            
            # 4. Bias correction
            m_corr = 1.0 / (1.0 - b1**self.t)
            v_corr = 1.0 / (1.0 - b2**self.t)
            
            # 5. Compute update
            # step = lr * (m / (1-b1^t)) / (sqrt(v / (1-b2^t)) + eps)
            m_hat = self.m[i].data.multiply_scalar(m_corr)
            v_hat = self.v[i].data.multiply_scalar(v_corr)
            
            denom = pb.sqrt(v_hat).add_scalar(self.eps)
            update = pb.divide(m_hat, denom).multiply_scalar(self.lr)
            
            # 6. Apply update
            p.data = pb.sub(p.data, update)
