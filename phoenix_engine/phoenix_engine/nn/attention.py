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
        # x shape: [B, T, C]
        B, T, C = x.shape
        
        # 1. Project Q, K, V -> [B, T, C]
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # 2. Split heads and permute: [B, num_heads, T, head_dim]
        q = q.view(B, T, self.num_heads, self.head_dim).permute([0, 2, 1, 3])
        k = k.view(B, T, self.num_heads, self.head_dim).permute([0, 2, 1, 3])
        v = v.view(B, T, self.num_heads, self.head_dim).permute([0, 2, 1, 3])
        
        # 3. KV-Cache (Conceptual - requires 'concat' op implementation in C++)
        if self.use_kv_cache:
            if self.k_cache is not None and self.v_cache is not None:
                # k = concat([self.k_cache, k], dim=2)
                # v = concat([self.v_cache, v], dim=2)
                pass
            self.k_cache = k
            self.v_cache = v
            
        # 4. Scaled Dot-Product Attention: softmax(Q @ K^T / sqrt(d)) @ V
        # Transpose K's last two dimensions (O(1) memory view)
        k_t = k.transpose(2, 3)
        
        # Batched Matrix Multiplication: [B, num_heads, T, T]
        attn_scores = q.matmul(k_t)
        
        # Apply Softmax (Scale by sqrt(d) omitted temporarily pending scalar division op)
        attn_probs = attn_scores.softmax()
        
        # Multiply by V: [B, num_heads, T, head_dim]
        attn_output = attn_probs.matmul(v)
        
        # 5. Concatenate heads back and project
        # Permute back: [B, T, num_heads, head_dim]
        # Must call .contiguous() before view because permute makes it non-contiguous!
        attn_output = attn_output.permute([0, 2, 1, 3]).contiguous()
        
        # Flatten heads: [B, T, C]
        attn_output = attn_output.view(B, T, C)
        
        # Final output projection
        return self.out_proj(attn_output)
