import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from phoenix_engine import Tensor, nn, optim

def test_adamw():
    print("Initializing AdamW Test...")
    
    # 1. Create a simple linear layer
    linear = nn.Linear(3, 2, bias=True)
    linear.weight.requires_grad = True
    linear.bias.requires_grad = True
        
    # 2. Input data [1, 3]
    x = Tensor.randn(1, 3, requires_grad=True)
    
    # 3. Optimizer
    optimizer = optim.AdamW([linear.weight, linear.bias], lr=0.1)
    
    # 4. Training loop (simple)
    print("Starting optimization loop...")
    for i in range(5):
        optimizer.zero_grad()
        y = linear(x)
        loss = y.sum()
        loss.backward()
        
        old_val = linear.weight.data.to_float_list()[0]
        optimizer.step()
        new_val = linear.weight.data.to_float_list()[0]
        
        print(f"Step {i+1}: Loss = {loss.item():.4f}, Weight[0] = {new_val:.4f} (diff: {new_val - old_val:.4f})")

    print("\nAdamW test completed successfully!")

if __name__ == "__main__":
    test_adamw()
