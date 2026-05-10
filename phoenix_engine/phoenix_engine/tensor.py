import _phoenix_backend as pb
from typing import List, Tuple, Optional, Callable, Union

class Tensor:
    def __init__(self, data: Union[pb.TensorData, List, float], requires_grad: bool = False, _grad_fn: Optional[Callable] = None, _prev: Tuple = ()):
        self.requires_grad = requires_grad
        self.grad: Optional['Tensor'] = None
        self._backward = lambda: None
        self._prev = set(_prev)
        
        if isinstance(data, pb.TensorData):
            self.data = data
        else:
            raise NotImplementedError("Initializing Tensor directly from list/float not yet implemented. Use ops or randn.")

    @staticmethod
    def randn(*shape: int, requires_grad: bool = False) -> 'Tensor':
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = shape[0]
        data = pb.randn(list(shape))
        return Tensor(data, requires_grad=requires_grad)

    @staticmethod
    def zeros(*shape: int, requires_grad: bool = False) -> 'Tensor':
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = shape[0]
        data = pb.zeros(list(shape))
        return Tensor(data, requires_grad=requires_grad)

    @staticmethod
    def ones(*shape: int, requires_grad: bool = False) -> 'Tensor':
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = shape[0]
        data = pb.ones(list(shape))
        return Tensor(data, requires_grad=requires_grad)
            
    @property
    def shape(self):
        return self.data.shape()

    def view(self, *shape: int) -> 'Tensor':
        out_data = self.data.view(list(shape))
        out = Tensor(out_data, _prev=(self,))
        
        def _backward():
            if self.requires_grad:
                grad_reshaped = out.grad.view(*self.shape)
                self.grad = self.grad + grad_reshaped if self.grad is not None else grad_reshaped
                
        out._backward = _backward
        return out

    def transpose(self, dim0: int, dim1: int) -> 'Tensor':
        out_data = self.data.transpose(dim0, dim1)
        out = Tensor(out_data, _prev=(self,))
        
        def _backward():
            if self.requires_grad:
                # pass grad backward via inverse transpose
                grad_T = out.grad.transpose(dim0, dim1)
                self.grad = self.grad + grad_T if self.grad is not None else grad_T
        
        out._backward = _backward
        return out

    def permute(self, dims: List[int]) -> 'Tensor':
        out_data = self.data.permute(dims)
        out = Tensor(out_data, _prev=(self,))
        
        def _backward():
            if self.requires_grad:
                # Inverse permutation
                inv_dims = [0] * len(dims)
                for i, d in enumerate(dims):
                    inv_dims[d] = i
                grad_perm = out.grad.permute(inv_dims)
                self.grad = self.grad + grad_perm if self.grad is not None else grad_perm
                
        out._backward = _backward
        return out

    def contiguous(self) -> 'Tensor':
        out_data = self.data.contiguous()
        out = Tensor(out_data, _prev=(self,))
        
        def _backward():
            if self.requires_grad:
                self.grad = self.grad + out.grad if self.grad is not None else out.grad
        
        out._backward = _backward
        return out
        
    @property
    def dtype(self):
        return self.data.dtype()
        
    @property
    def device(self):
        return self.data.device()

    def backward(self):
        # Topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        if self.grad is None:
            self.grad = Tensor.ones(*self.shape)

        for v in reversed(topo):
            v._backward()

    def __add__(self, other: 'Tensor') -> 'Tensor':
        out_data = pb.add(self.data, other.data)
        out = Tensor(out_data, _prev=(self, other))

        def _backward():
            if self.requires_grad:
                self.grad = self.grad + out.grad if self.grad is not None else out.grad
            if other.requires_grad:
                other.grad = other.grad + out.grad if other.grad is not None else out.grad

        out._backward = _backward
        return out

    def __mul__(self, other: 'Tensor') -> 'Tensor':
        out_data = pb.multiply(self.data, other.data)
        out = Tensor(out_data, _prev=(self, other))

        def _backward():
            if self.requires_grad:
                grad_self = other * out.grad
                self.grad = self.grad + grad_self if self.grad is not None else grad_self
            if other.requires_grad:
                grad_other = self * out.grad
                other.grad = other.grad + grad_other if other.grad is not None else grad_other

        out._backward = _backward
        return out

    def __sub__(self, other: 'Tensor') -> 'Tensor':
        out_data = pb.sub(self.data, other.data)
        out = Tensor(out_data, _prev=(self, other))

        def _backward():
            if self.requires_grad:
                self.grad = self.grad + out.grad if self.grad is not None else out.grad
            if other.requires_grad:
                # Subtraction requires negative gradient for the second operand
                # We don't have unary minus yet, so we emulate it with out.grad * (-1)
                # But we don't have scalar mul yet either! 
                # Let's add a quick hack if needed or implement scalar mul.
                pass

        out._backward = _backward
        return out

    def sum(self) -> 'Tensor':
        out_data = pb.sum(self.data)
        out = Tensor(out_data, _prev=(self,))

        def _backward():
            if self.requires_grad:
                pass

        out._backward = _backward
        return out

    def relu(self) -> 'Tensor':
        out_data = pb.relu(self.data)
        out = Tensor(out_data, _prev=(self,))

        def _backward():
            if self.requires_grad:
                pass

        out._backward = _backward
        return out

    def matmul(self, other: 'Tensor') -> 'Tensor':
        out_data = pb.gemm(self.data, other.data)
        out = Tensor(out_data, _prev=(self, other))
        
        def _backward():
            if self.requires_grad:
                # self.grad += out.grad @ other.data.T
                grad_self = out.grad.matmul(other.transpose(-2, -1))
                self.grad = self.grad + grad_self if self.grad is not None else grad_self
            if other.requires_grad:
                # other.grad += self.data.T @ out.grad
                grad_other = self.transpose(-2, -1).matmul(out.grad)
                other.grad = other.grad + grad_other if other.grad is not None else grad_other
                
        out._backward = _backward
        return out

    def softmax(self) -> 'Tensor':
        out_data = pb.softmax(self.data)
        out = Tensor(out_data, _prev=(self,))

        def _backward():
            if self.requires_grad:
                pass

        out._backward = _backward
        return out

    def layernorm(self, weight: Optional['Tensor'] = None, bias: Optional['Tensor'] = None, eps: float = 1e-5) -> 'Tensor':
        w_data = weight.data if weight is not None else None
        b_data = bias.data if bias is not None else None
        
        out_data = pb.layernorm(self.data, w_data, b_data, eps)
        
        prevs = [self]
        if weight is not None: prevs.append(weight)
        if bias is not None: prevs.append(bias)
        
        out = Tensor(out_data, _prev=tuple(prevs))

        def _backward():
            if self.requires_grad:
                pass
            if weight is not None and weight.requires_grad:
                pass
            if bias is not None and bias.requires_grad:
                pass

        out._backward = _backward
        return out

    def __repr__(self):
        return f"Tensor(shape={self.shape}, requires_grad={self.requires_grad})"
