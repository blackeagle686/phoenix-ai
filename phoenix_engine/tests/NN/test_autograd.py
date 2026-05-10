import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from phoenix_engine import Tensor, nn, optim

def test_autograd():
    print("Initializing Autograd Test...")
    
    # 1. Create a simple linear layer
    linear = nn.Linear(3, 2, bias=True)
    
    # Enable gradients for parameters
    linear.weight.requires_grad = True
    if linear.bias is not None:
        linear.bias.requires_grad = True
        
    print(f"Weight shape: {linear.weight.shape}")
    if linear.bias is not None:
        print(f"Bias shape: {linear.bias.shape}")
        
    # 2. Input data [1, 3]
    x = Tensor.randn(1, 3, requires_grad=True)
    
    # 3. Forward Pass
    # x: [1, 3], weight: [3, 2] -> out: [1, 2]
    y = linear(x)
    
    # 4. Dummy Loss (Sum)
    loss = y.sum()
    print(f"Loss forward pass complete. Loss shape: {loss.shape}")
    
    # 5. Backward Pass
    loss.backward()
    
    print("\nBackward pass completed successfully!")
    print(f"Input grad exists: {x.grad is not None}")
    print(f"Weight grad exists: {linear.weight.grad is not None}")
    if linear.bias is not None:
        print(f"Bias grad exists: {linear.bias.grad is not None}")
        
    # 6. Optimizer Step
    optimizer = optim.SGD([linear.weight, linear.bias], lr=0.1)
    
    # Save old weight
    old_weight_val = linear.weight.data
    
    optimizer.step()
    print("Optimizer step completed!")
    
    # Check that data pointer was updated
    print(f"Weight data object updated by optimizer: {old_weight_val is not linear.weight.data}")

if __name__ == "__main__":
    test_autograd()
