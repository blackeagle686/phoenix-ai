from .module import Module
from ..tensor import Tensor
import _phoenix_backend as pb

class MSELoss(Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        diff = pred - target
        sq_diff = diff * diff
        return sq_diff.sum()

class CrossEntropyLoss(Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        # logits: [batch, vocab_size], target: [batch] (int32)
        out_data = pb.softmax_cross_entropy(logits.data, target.data)
        out = Tensor(out_data, requires_grad=logits.requires_grad, _prev=(logits, target))
        
        def _backward():
            if logits.requires_grad:
                # Use fused backward
                grad_data = pb.softmax_cross_entropy_backward(out.grad.data, logits.data, target.data)
                logits.grad = Tensor(grad_data) if logits.grad is None else logits.grad + Tensor(grad_data)
                
        out._backward = _backward
        # Mean reduction for loss
        return out.sum() * (1.0 / target.shape[0])
