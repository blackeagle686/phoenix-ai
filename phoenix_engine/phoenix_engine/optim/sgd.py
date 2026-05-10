from ..nn.module import Parameter
from typing import List

class SGD:
    def __init__(self, params: List[Parameter], lr: float = 0.01):
        self.params = params
        self.lr = lr

    def step(self):
        for p in self.params:
            if p.grad is not None:
                # p.data = p.data - self.lr * p.grad.data
                # For now, we need to implement the actual subtraction in C++ or a dedicated op
                pass

    def zero_grad(self):
        for p in self.params:
            p.grad = None
