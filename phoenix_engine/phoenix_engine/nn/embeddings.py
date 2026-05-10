from .module import Module, Parameter
from ..tensor import Tensor

class Embedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        # Xavier/Uniform initialization for embeddings
        self.weight = Parameter(Tensor.randn(num_embeddings, embedding_dim))
        
    def forward(self, x: Tensor) -> Tensor:
        # x is a tensor of indices (Int32)
        return x.embedding(self.weight)
