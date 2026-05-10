import os
import sys

# 1. Resolve paths
# Root of phoenix_engine
engine_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# The folder containing the actual python code
python_src = os.path.join(engine_root, 'python')

# 2. Add paths to sys.path
sys.path.append(engine_root)
sys.path.append(python_src)

# 3. Handle the 'phoenix_engine' package alias
try:
    from phoenix_engine import Tensor
    from phoenix_engine.nn import Linear, MSELoss
    from phoenix_engine.optim import SGD
except ImportError:
    # Fallback for when not installed via pip: import directly from the python folder
    print("Package not installed, falling back to direct imports from 'python/' folder...")
    import tensor as phoenix_engine
    from tensor import Tensor
    import nn
    from nn import Linear, MSELoss
    import optim
    from optim import SGD

try:
    import _phoenix_backend as pb
except ImportError:
    print("\n[ERROR] _phoenix_backend not found!")
    print("On Colab, you MUST compile the backend first. Run:")
    print("!cd /content/phoenix-ai/phoenix_engine && pip install -e .")
    sys.exit(1)

def test_nn_training():
    print("Initializing Phoenix-Engine Training Test...")
    
    # 1. Create dummy data (10 samples, 2 features)
    # y = 2*x1 + 3*x2 + 1
    # For simplicity, we just use random for now since we don't have numpy-to-tensor yet
    x = Tensor.randn([10, 2])
    target = Tensor.randn([10, 1]) # Random target for demonstration
    
    # 2. Define Model
    model = Linear(2, 1)
    print(f"Model created: {model}")
    
    # 3. Setup Criterion and Optimizer
    criterion = MSELoss()
    optimizer = SGD(model.parameters(), lr=0.01)
    
    # 4. Training Loop
    print("\nStarting Training Loop...")
    for epoch in range(10):
        optimizer.zero_grad()
        
        # Forward Pass
        pred = model(x)
        
        # Calculate Loss
        loss = criterion(pred, target)
        
        # Backward Pass
        loss.backward()
        
        # Optimization Step
        optimizer.step()
        
        print(f"Epoch {epoch+1}/10 - Loss: {loss}")

    print("\nTraining test complete!")

if __name__ == "__main__":
    try:
        test_nn_training()
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
