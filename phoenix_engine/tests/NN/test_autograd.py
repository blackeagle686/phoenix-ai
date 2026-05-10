import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from phoenix_engine import Tensor, nn, optim

import _phoenix_backend as pb
class LocalSGD:
    def __init__(self, parameters, lr=0.01):
        self.parameters = parameters
        self.lr = lr
    def step(self):
        for p in self.parameters:
            if p.grad is not None:
                step = p.grad * self.lr
                p.data = pb.sub(p.data, step.data)
    def zero_grad(self):
        for p in self.parameters: p.grad = None

def test_autograd():
    print("Initializing Autograd Test...")
    
    # 1. Create a simple linear layer
    linear = nn.Linear(3, 2, bias=True)
    
    # Enable gradients for parameters
    linear.weight.requires_grad = True
    if linear.bias is not None:
        linear.bias.requires_grad = True
        
    # 2. Input data [1, 3]
    x = Tensor.randn(1, 3, requires_grad=True)
    
    # 3. Forward Pass
    y = linear(x)
    
    # 4. Dummy Loss (Sum)
    loss = y.sum()
    print(f"Loss forward pass complete. Loss value: {loss.item()}")
    
    # 5. Backward Pass
    loss.backward()
    
    print("\nBackward pass completed successfully!")
    print(f"Input grad shape: {x.grad.shape}")
    print(f"Weight grad shape: {linear.weight.grad.shape}")
    if linear.bias is not None:
        print(f"Bias grad shape: {linear.bias.grad.shape}")
        
    # 6. Optimizer Step
    optimizer = LocalSGD([linear.weight, linear.bias], lr=0.1)
    
    # Save old weight pointer for comparison
    old_weight_data = linear.weight.data
    print(f"DEBUG: Before step, weight data ID: {id(old_weight_data)}")
    
    optimizer.step()
    print(f"DEBUG: After step, weight data ID: {id(linear.weight.data)}")
    print("Optimizer step completed!")
    
    # Verify that the weight was updated (different data object)
    if linear.weight.data is not old_weight_data:
        print("SUCCESS: Weight data updated by optimizer.")
    else:
        print("FAILED: Weight data NOT updated by optimizer.")
        
    # Check if we can zero gradients
    optimizer.zero_grad()
    if linear.weight.grad is None:
        print("SUCCESS: Gradients zeroed.")
    else:
        print("FAILED: Gradients not zeroed.")

if __name__ == "__main__":
    test_autograd()
