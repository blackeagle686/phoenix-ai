import _phoenix_backend as pb
from typing import List, Tuple, Optional, Callable, Union

class Tensor:
    def __init__(self, data: Union[pb.TensorData, 'Tensor', List, float], requires_grad: bool = False, _grad_fn: Optional[Callable] = None, _prev: Tuple = ()):
        if isinstance(data, Tensor):
            data = data.data
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
            
    def item(self) -> float:
        return self.data.to_float_list()[0]

    @staticmethod
    def long(data: List[int], shape: Optional[List[int]] = None) -> 'Tensor':
        if shape is None:
            shape = [len(data)]
        data_obj = pb.from_list_int32(data, shape)
        return Tensor(data_obj, requires_grad=False)

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
        # Handle negative indices
        ndims = len(self.shape)
        if dim0 < 0: dim0 += ndims
        if dim1 < 0: dim1 += ndims
        
        out_data = self.data.transpose(dim0, dim1)
        out = Tensor(out_data, requires_grad=self.requires_grad, _prev=(self,))
        
        def _backward():
            if self.requires_grad:
                # pass grad backward via inverse transpose
                grad_T = out.grad.transpose(dim0, dim1)
                self.grad = self.grad + grad_T if self.grad is not None else grad_T
        
        out._backward = _backward
        return out

    def permute(self, dims: List[int]) -> 'Tensor':
        # Handle negative indices
        ndims = len(self.shape)
        actual_dims = [(d + ndims if d < 0 else d) for d in dims]
        
        out_data = self.data.permute(actual_dims)
        out = Tensor(out_data, requires_grad=self.requires_grad, _prev=(self,))
        
        def _backward():
            if self.requires_grad:
                # Inverse permutation
                inv_dims = [0] * len(actual_dims)
                for i, d in enumerate(actual_dims):
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

    def __add__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        if isinstance(other, (float, int)):
            out_data = pb.add_scalar(self.data, float(other))
            out = Tensor(out_data, requires_grad=self.requires_grad, _prev=(self,))
            def _backward():
                if self.requires_grad:
                    self.grad = self.grad + out.grad if self.grad is not None else out.grad
            out._backward = _backward
            return out
        else:
            out_data = pb.add(self.data, other.data)
            out = Tensor(out_data, requires_grad=(self.requires_grad or other.requires_grad), _prev=(self, other))

            def _backward():
                if self.requires_grad:
                    self.grad = self.grad + out.grad if self.grad is not None else out.grad
                if other.requires_grad:
                    other.grad = other.grad + out.grad if other.grad is not None else out.grad

            out._backward = _backward
            return out

    def __radd__(self, other: Union[float, int]) -> 'Tensor':
        return self + other

    def __mul__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        if isinstance(other, (float, int)):
            out_data = pb.multiply_scalar(self.data, float(other))
            out = Tensor(out_data, requires_grad=self.requires_grad, _prev=(self,))
            def _backward():
                if self.requires_grad:
                    grad_self = out.grad * other
                    self.grad = self.grad + grad_self if self.grad is not None else grad_self
            out._backward = _backward
            return out
        elif getattr(other, 'shape', None) == [1]:
            # Treat single-element tensor as a scalar for broadcasting
            return self * other.item()
        else:
            out_data = pb.multiply(self.data, other.data)
            out = Tensor(out_data, requires_grad=(self.requires_grad or other.requires_grad), _prev=(self, other))
            def _backward():
                if self.requires_grad:
                    grad_self = other * out.grad
                    self.grad = self.grad + grad_self if self.grad is not None else grad_self
                if other.requires_grad:
                    grad_other = self * out.grad
                    other.grad = other.grad + grad_other if other.grad is not None else grad_other
            out._backward = _backward
            return out

    def __rmul__(self, other: Union[float, int]) -> 'Tensor':
        return self * other

    def __sub__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        if isinstance(other, (float, int)):
            return self + (-float(other))
        else:
            out_data = pb.sub(self.data, other.data)
            out = Tensor(out_data, requires_grad=(self.requires_grad or other.requires_grad), _prev=(self, other))

            def _backward():
                if self.requires_grad:
                    self.grad = self.grad + out.grad if self.grad is not None else out.grad
                if other.requires_grad:
                    grad_other = out.grad * (-1.0)
                    other.grad = other.grad + grad_other if other.grad is not None else grad_other

            out._backward = _backward
            return out

    def __truediv__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        if isinstance(other, (float, int)):
            out_data = pb.divide_scalar(self.data, float(other))
            out = Tensor(out_data, requires_grad=self.requires_grad, _prev=(self,))
            def _backward():
                if self.requires_grad:
                    # grad = out.grad / other
                    grad_self = out.grad * (1.0 / other)
                    self.grad = self.grad + grad_self if self.grad is not None else grad_self
            out._backward = _backward
            return out
        else:
            out_data = pb.divide(self.data, other.data)
            out = Tensor(out_data, requires_grad=(self.requires_grad or other.requires_grad), _prev=(self, other))
            def _backward():
                if self.requires_grad:
                    # grad_a = out.grad / b
                    grad_self = out.grad / other
                    self.grad = self.grad + grad_self if self.grad is not None else grad_self
                if other.requires_grad:
                    # grad_b = -out.grad * a / b^2
                    grad_other = (out.grad * self * -1.0) / (other * other)
                    other.grad = other.grad + grad_other if other.grad is not None else grad_other
            out._backward = _backward
            return out

    def sum(self) -> 'Tensor':
        out_data = pb.sum(self.data)
        out = Tensor(out_data, requires_grad=self.requires_grad, _prev=(self,))

        def _backward():
            if self.requires_grad:
                # Propagate gradient to all elements
                grad_self = Tensor.ones(*self.shape) * out.grad
                self.grad = self.grad + grad_self if self.grad is not None else grad_self

        out._backward = _backward
        return out

    def relu(self) -> 'Tensor':
        out_data = pb.relu(self.data)
        out = Tensor(out_data, requires_grad=self.requires_grad, _prev=(self,))

        def _backward():
            if self.requires_grad:
                # grad = out.grad * (self > 0)
                # We need a masked_fill or something, but for now we can just use relu logic or a quick hack
                # Actually, we don't have > 0 op yet. 
                # Let's skip for now or implement a basic one if needed.
                pass

        out._backward = _backward
        return out

    def matmul(self, other: 'Tensor') -> 'Tensor':
        out_data = pb.gemm(self.data, other.data)
        out = Tensor(out_data, requires_grad=(self.requires_grad or other.requires_grad), _prev=(self, other))
        
        def _backward():
            if self.requires_grad:
                grad_self = out.grad.matmul(other.transpose(-2, -1))
                self.grad = self.grad + grad_self if self.grad is not None else grad_self
            if other.requires_grad:
                grad_other = self.transpose(-2, -1).matmul(out.grad)
                other.grad = other.grad + grad_other if other.grad is not None else grad_other
                
        out._backward = _backward
        return out

    def softmax(self) -> 'Tensor':
        out_data = pb.softmax(self.data)
        out = Tensor(out_data, requires_grad=self.requires_grad, _prev=(self,))

        def _backward():
            if self.requires_grad:
                # Softmax backward is complex: s = softmax(x); grad = s * (dz - sum(dz * s))
                # We'll need a few more ops or a fused C++ kernel
                pass

        out._backward = _backward
        return out

    def sqrt(self) -> 'Tensor':
        out_data = pb.sqrt(self.data)
        out = Tensor(out_data, requires_grad=self.requires_grad, _prev=(self,))

        def _backward():
            if self.requires_grad:
                # grad = out.grad * (0.5 / sqrt(self)) = out.grad * 0.5 / out
                grad_self = out.grad * 0.5 * (1.0 / self.item_pow(-0.5)) # This is getting complex, let's just use 0.5 / out
                # Actually, 1.0 / out works if out is already sqrt(self)
                # But we don't have 1.0 / Tensor yet!
                pass

        out._backward = _backward
        return out
    def embedding(self, weight: 'Tensor') -> 'Tensor':
        out_data = pb.embedding_forward(self.data, weight.data)
        out = Tensor(out_data, requires_grad=weight.requires_grad, _prev=(self, weight))

        def _backward():
            if weight.requires_grad:
                # weight.grad[indices] += out.grad
                # This is a scatter_add operation.
                # For now, we'll need a C++ kernel for embedding_backward or a scatter_add.
                pass

        out._backward = _backward
        return out

    def layernorm(self, weight: Optional['Tensor'] = None, bias: Optional['Tensor'] = None, eps: float = 1e-5) -> 'Tensor':
        w_data = weight.data if weight is not None else None
        b_data = bias.data if bias is not None else None
        
        requires_grad = self.requires_grad or (weight.requires_grad if weight else False) or (bias.requires_grad if bias else False)
        out_data = pb.layernorm(self.data, w_data, b_data, eps)
        
        prevs = [self]
        if weight is not None: prevs.append(weight)
        if bias is not None: prevs.append(bias)
        
        out = Tensor(out_data, requires_grad=requires_grad, _prev=tuple(prevs))

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
