"""Simple transformer model using NSA attention."""

import torch
import torch.nn as nn

from .attention import NSAAttention
from .config import NSAConfig


class NSATransformerBlock(nn.Module):
    """Transformer block using NSA attention."""
    
    def __init__(self, config: NSAConfig):
        super().__init__()
        self.attention = NSAAttention(config)
        self.norm1 = nn.LayerNorm(config.hidden_dim)
        self.norm2 = nn.LayerNorm(config.hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * 4),
            nn.GELU(),
            nn.Linear(config.hidden_dim * 4, config.hidden_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm architecture
        normed = self.norm1(x)
        x = x + self.attention(normed)
        normed = self.norm2(x)
        x = x + self.mlp(normed)
        return x


class NSATransformer(nn.Module):
    """Simple transformer using NSA attention."""
    
    def __init__(
        self,
        vocab_size: int,
        num_layers: int = 2,
        hidden_dim: int = 256,
        num_heads: int = 8,
        head_dim: int = 32
    ):
        super().__init__()
        
        # Create config with model hyperparameters
        self.config = NSAConfig(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            compression_block_size=64,  # l in paper
            compression_stride=32,      # d in paper
            selection_block_size=64,    # l' in paper
            window_size=256,           # w in paper
            num_selected_blocks=8      # n in paper
        )
        
        # Layers
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        max_seq_len = 1024
        self.pos_embedding = nn.Parameter(
            torch.randn(1, max_seq_len, hidden_dim) / (hidden_dim ** 0.5)
        )
        self.layers = nn.ModuleList([
            NSATransformerBlock(self.config)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L = x.shape
        
        # Embeddings
        h = self.embedding(x)
        h = h + self.pos_embedding[:, :L]
        
        # Transformer layers
        for layer in self.layers:
            h = layer(h)
        
        # Output
        h = self.norm(h)
        return self.head(h)
