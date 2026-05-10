import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from phoenix_engine import Tensor, nn, optim
import _phoenix_backend as pb

class LocalAdamW:
    def __init__(self, parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        self.parameters = parameters
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.m = [Tensor.zeros(*p.shape) for p in parameters]
        self.v = [Tensor.zeros(*p.shape) for p in parameters]
    def zero_grad(self):
        for p in self.parameters: p.grad = None
    def step(self):
        self.t += 1
        b1, b2 = self.betas
        for i, p in enumerate(self.parameters):
            if p.grad is None: continue
            if self.weight_decay != 0:
                decay_step = p.data.multiply_scalar(self.lr * self.weight_decay)
                p.data = pb.sub(p.data, decay_step)
            m_term1 = self.m[i].data.multiply_scalar(b1)
            m_term2 = p.grad.data.multiply_scalar(1.0 - b1)
            self.m[i].data = pb.add(m_term1, m_term2)
            v_term1 = self.v[i].data.multiply_scalar(b2)
            grad_sq = pb.multiply(p.grad.data, p.grad.data)
            v_term2 = grad_sq.multiply_scalar(1.0 - b2)
            self.v[i].data = pb.add(v_term1, v_term2)
            m_corr = 1.0 / (1.0 - b1**self.t)
            v_corr = 1.0 / (1.0 - b2**self.t)
            m_hat = self.m[i].data.multiply_scalar(m_corr)
            v_hat = self.v[i].data.multiply_scalar(v_corr)
            denom = pb.sqrt(v_hat).add_scalar(self.eps)
            update = pb.divide(m_hat, denom).multiply_scalar(self.lr)
            p.data = pb.sub(p.data, update)

def test_adamw():
    print("Initializing AdamW Test...")
    
    # 1. Create a simple linear layer
    linear = nn.Linear(3, 2, bias=True)
    linear.weight.requires_grad = True
    linear.bias.requires_grad = True
        
    # 2. Input data [1, 3]
    x = Tensor.randn(1, 3, requires_grad=True)
    
    # 3. Optimizer
    optimizer = LocalAdamW([linear.weight, linear.bias], lr=0.1)
    
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
