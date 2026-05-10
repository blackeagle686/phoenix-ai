from .module import Module
from ..tensor import Tensor

class MSELoss(Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        diff = pred - target
        sq_diff = diff * diff
        return sq_diff.sum()

    def __call__(self, pred: Tensor, target: Tensor) -> Tensor:
        return self.forward(pred, target)
