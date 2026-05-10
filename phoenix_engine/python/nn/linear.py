from .module import Module, Parameter
from ..tensor import Tensor
import math

class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize weights using randn (simple initialization for now)
        # In a real framework, we'd use Xavier or He initialization
        self.weight = Parameter(Tensor.randn([in_features, out_features]).data)
        
        if bias:
            self.bias = Parameter(Tensor.randn([1, out_features]).data)
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        out = x.matmul(self.weight)
        if self.bias is not None:
            out = out + self.bias
        return out

    def __repr__(self):
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None})"




from phoenix_engine import Tensor
from phoenix_engine.nn import Module, Linear, MSELoss
from phoenix_engine.optim import SGD

class SimpleNet(Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(10, 5)
        self.fc2 = Linear(5, 1)

    def forward(self, x):
        x = self.fc1(x).relu()
        return self.fc2(x)

model = SimpleNet()
criterion = MSELoss()
optimizer = SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(100):
    optimizer.zero_grad()
    
    # Forward pass (Fast C++ execution)
    pred = model(input_tensor)
    loss = criterion(pred, target_tensor)
    
    # Backward pass (Autograd engine)
    loss.backward()
    
    # Step (Optimized updates)
    optimizer.step()
