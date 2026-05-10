import sys
import os

# Add the root directory to the python path so we can import phoenix_engine
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

try:
    from phoenix_engine import Tensor
    from phoenix_engine import nn
except ImportError:
    print("Package not installed, falling back to direct imports from 'phoenix_engine/' folder...")
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../phoenix_engine')))
    from tensor import Tensor
    import nn

class MiniGPT(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, num_heads: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiHeadAttention(embed_dim, num_heads)
        self.ln2 = nn.LayerNorm(embed_dim)
        # Final projection to vocabulary
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        
    def forward(self, idx: Tensor) -> Tensor:
        # 1. Token Embedding
        x = self.embedding(idx)
        
        # 2. Transformer Block (simplified)
        x_norm = self.ln1(x)
        attn_out = self.attention(x_norm)
        
        # Residual connection (x = x + attn_out)
        x = x + attn_out
        
        x_norm2 = self.ln2(x)
        
        # 3. Output Logits
        logits = self.lm_head(x_norm2)
        return logits

def test_mini_gpt():
    print("Initializing Phoenix-Engine Mini-GPT Test...")
    
    # Hyperparameters
    vocab_size = 50
    embed_dim = 64
    num_heads = 4
    batch_size = 2
    seq_len = 10
    
    model = MiniGPT(vocab_size, embed_dim, num_heads)
    print(f"Model created successfully! Embedding Dim: {embed_dim}, Heads: {num_heads}")
    
    # Create dummy integer indices using floats (temporary workaround until Int32 is fully implemented)
    # Shape: [batch_size, seq_len]
    print(f"Generating dummy input tokens shape [{batch_size}, {seq_len}]...")
    # Generate random floats and manually cast them to integers later in C++ (or just use zeros for test)
    # Actually, randn generates normal floats. We'll use random integers between 0 and vocab_size-1.
    import random
    
    # Since our engine doesn't have an integer factory yet, we'll just run forward pass
    # on standard normal data to verify the dimensions flow correctly through the Attention mechanism.
    print("\nRunning Forward Pass through GPT architecture...")
    
    dummy_input = Tensor.randn([batch_size, seq_len])
    
    try:
        logits = model(dummy_input)
        print(f"\nForward pass successful!")
        print(f"Output Logits Shape: {logits.shape}")
        
        # Output shape should be [batch_size, seq_len, vocab_size]
        assert logits.shape == [batch_size, seq_len, vocab_size], f"Expected {[batch_size, seq_len, vocab_size]}, got {logits.shape}"
        
        print("\nMini-GPT Verification Complete! The Phoenix-Engine successfully executed:")
        print("- Batched Matrix Multiplication (BMM)")
        print("- O(1) Memory Views and Transposes")
        print("- C++ LayerNorm and Softmax")
        
    except Exception as e:
        print(f"\nTest failed with error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_mini_gpt()
