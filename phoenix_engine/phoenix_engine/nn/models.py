from .module import Module, Parameter
from .linear import Linear
from .normalization import LayerNorm
from .embeddings import Embedding
from .attention import MultiHeadAttention
from ..tensor import Tensor
import math

class MLP(Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.c_fc = Linear(embed_dim, 4 * embed_dim)
        self.c_proj = Linear(4 * embed_dim, embed_dim)
        
    def forward(self, x: Tensor) -> Tensor:
        x = self.c_fc(x).relu()
        x = self.c_proj(x)
        return x

class Block(Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.ln_1 = LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.ln_2 = LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim)
        
    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(Module):
    def __init__(self, vocab_size: int, embed_dim: int, num_heads: int, num_layers: int, max_seq_len: int):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.token_embedding = Embedding(vocab_size, embed_dim)
        self.position_embedding = Parameter(Tensor.randn(max_seq_len, embed_dim))
        
        self.blocks = [Block(embed_dim, num_heads) for _ in range(num_layers)]
        # Register blocks as submodules
        for i, b in enumerate(self.blocks):
            self._modules[f"block_{i}"] = b
            
        self.ln_f = LayerNorm(embed_dim)
        self.lm_head = Linear(embed_dim, vocab_size, bias=False)
        
    def forward(self, idx: Tensor) -> Tensor:
        # idx: [B, T]
        B, T = idx.shape
        assert T <= self.max_seq_len, f"Cannot forward sequence of length {T}, max is {self.max_seq_len}"
        
        # Token and position embeddings
        tok_emb = self.token_embedding(idx) # [B, T, C]
        pos_emb = self.position_embedding.view(1, self.max_seq_len, -1) # Broad-castable
        # Crop position embedding to actual length T
        # (Need slicing support for this, or just take the first T rows)
        # For now, let's assume we use full max_seq_len or add a slice kernel
        x = tok_emb + pos_emb.view(1, T, -1) if T == self.max_seq_len else tok_emb # Placeholder
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
            
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits
