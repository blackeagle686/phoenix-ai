import math
from typing import Optional, Tuple
from ..tensor import Tensor
from .module import Module, Parameter
from .linear import Linear

class MultiHeadAttention(Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0, use_kv_cache: bool = False):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.use_kv_cache = use_kv_cache
        
        # Projection matrices
        self.q_proj = Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = Linear(embed_dim, embed_dim, bias=False)
        
        # KV Cache state
        self.k_cache: Optional[Tensor] = None
        self.v_cache: Optional[Tensor] = None
        
    def reset_cache(self):
        """Clears the KV cache for a new generation sequence."""
        self.k_cache = None
        self.v_cache = None

    def forward(self, x: Tensor) -> Tensor:
        # x shape: [batch_size, seq_len, embed_dim]
        # TODO: Implement full attention calculation
        # 1. Project Q, K, V
        # 2. Split heads and permute: [batch, num_heads, seq_len, head_dim]
        # 3. Handle KV-Cache concatenation
        # 4. Scaled Dot-Product: softmax(Q @ K^T / sqrt(d)) @ V
        # 5. Concatenate heads and out_proj
        pass
