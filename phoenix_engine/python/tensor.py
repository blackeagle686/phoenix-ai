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
            # For now, we assume data is already a TensorData object passed from ops
            # In a full implementation, we'd add logic to convert lists/numpy arrays to TensorData
            raise NotImplementedError("Initializing Tensor directly from list/float not yet implemented. Use ops.")
            
    @property
    def shape(self):
        return self.data.shape()
        
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
        # For simplicity in this initial version, we leave grad initialization to the user or assume scalar 1
        
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

    def __repr__(self):
        return f"Tensor(shape={self.shape}, requires_grad={self.requires_grad})"
