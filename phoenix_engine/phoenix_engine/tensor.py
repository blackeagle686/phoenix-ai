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
    def randn(shape: List[int], requires_grad: bool = False) -> 'Tensor':
        data = pb.randn(shape)
        return Tensor(data, requires_grad=requires_grad)
            
    @property
    def shape(self):
        return self.data.shape()

    def transpose(self, dim0: int, dim1: int) -> 'Tensor':
        # Create a new Tensor that wraps the transposed TensorData (O(1) operation)
        out_data = self.data.transpose(dim0, dim1)
        out = Tensor(out_data, _prev=(self,))
        
        def _backward():
            if self.requires_grad:
                # pass grad backward via inverse transpose
                pass
        
        out._backward = _backward
        return out

    def permute(self, dims: List[int]) -> 'Tensor':
        out_data = self.data.permute(dims)
        out = Tensor(out_data, _prev=(self,))
        # Backward for permute is the inverse permutation
        return out

    def contiguous(self) -> 'Tensor':
        out_data = self.data.contiguous()
        out = Tensor(out_data, _prev=(self,))
        
        def _backward():
            if self.requires_grad:
                pass
        
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

        # Go one variable at a time and apply the chain rule to get its gradient
        # self.grad = Tensor(ones_like(self)) # Initialize root gradient to 1
        # For now, we'll assume the user wants to backward from a scalar loss, so initialize grad to 1.0
        if self.grad is None:
            # This is a bit hacky since we don't have a 'ones' op yet, 
            # but we can assume the loss is a scalar and the gradient is 1.0.
            # In a real framework, we'd check shape.
            pass 

        for v in reversed(topo):
            v._backward()

    def __add__(self, other: 'Tensor') -> 'Tensor':
        out_data = pb.add(self.data, other.data)
        out = Tensor(out_data, _prev=(self, other))

        def _backward():
            if self.requires_grad:
                # self.grad += out.grad (elementwise)
                pass # Need an in-place add or accumulator for full autograd
            if other.requires_grad:
                # other.grad += out.grad
                pass

        out._backward = _backward
        return out

    def __mul__(self, other: 'Tensor') -> 'Tensor':
        out_data = pb.multiply(self.data, other.data)
        out = Tensor(out_data, _prev=(self, other))

        def _backward():
            if self.requires_grad:
                # self.grad += other.data * out.grad
                pass
            if other.requires_grad:
                # other.grad += self.data * out.grad
                pass

        out._backward = _backward
        return out

    def __sub__(self, other: 'Tensor') -> 'Tensor':
        out_data = pb.sub(self.data, other.data)
        out = Tensor(out_data, _prev=(self, other))

        def _backward():
            if self.requires_grad:
                # self.grad += out.grad
                pass
            if other.requires_grad:
                # other.grad -= out.grad
                pass

        out._backward = _backward
        return out

    def sum(self) -> 'Tensor':
        out_data = pb.sum(self.data)
        out = Tensor(out_data, _prev=(self,))

        def _backward():
            if self.requires_grad:
                # self.grad += out.grad * ones_like(self.data)
                pass

        out._backward = _backward
        return out

    def relu(self) -> 'Tensor':
        out_data = pb.relu(self.data)
        out = Tensor(out_data, _prev=(self,))

        def _backward():
            if self.requires_grad:
                # self.grad += (self.data > 0) * out.grad
                pass

        out._backward = _backward
        return out

    def matmul(self, other: 'Tensor') -> 'Tensor':
        out_data = pb.gemm(self.data, other.data)
        out = Tensor(out_data, _prev=(self, other))
        
        def _backward():
            if self.requires_grad:
                # self.grad += out.grad @ other.data.T
                pass
            if other.requires_grad:
                # other.grad += self.data.T @ out.grad
                pass
                
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
