"""Implements token compression for NSA attention."""

import torch
import torch.nn as nn
# import torch.nn.functional as F  # Will be needed for future extensions

from .config import NSAConfig


class TokenCompressor(nn.Module):
    """Compresses blocks of tokens into single representations."""

    def __init__(self, config: NSAConfig):
        super().__init__()
        self.config = config

        # Learnable compression MLP (φ function from paper)
        self.compress_net = nn.Sequential(
            nn.Linear(
                config.head_dim * config.compression_block_size,
                config.head_dim * 4
            ),
            nn.GELU(),
            nn.Linear(config.head_dim * 4, config.head_dim)
        )

        # Learnable intra-block positional encoding
        # Initialize with small values for stable training
        self.pos_embedding = nn.Parameter(
            torch.randn(1, config.compression_block_size, config.head_dim) / 
            (config.head_dim ** 0.5)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compresses blocks of tokens using learnable MLP with intra-block positional encoding.
        
        Following the paper's formulation:
        K̃ᶜᵐᵖₜ = φ(k_{i:i+l}) for i = 1, ..., t-l+1
        
        Args:
            x: (batch_size, num_heads, seq_len, head_dim)
        Returns:
            compressed: (batch_size, num_heads, compressed_len, head_dim)
        """
        B, H, L, D = x.shape
        block_size = self.config.compression_block_size
        stride = self.config.compression_stride

        # Following paper: partition input sequence into blocks
        # and compress each block using MLP with intra-block positional encoding
        blocks = []
        for i in range(0, L - block_size + 1, stride):
            # Extract block k_{i:i+l}
            block = x[:, :, i:i+block_size, :]  # (B, H, l, D)
            
            # Add intra-block positional encoding
            pos_emb = self.pos_embedding.view(1, 1, block_size, D)
            block = block + pos_emb  # Position-aware representation
            blocks.append(block)

        if not blocks:
            return x.new_zeros((B, H, 0, D))

        # Stack blocks for parallel processing: (B, H, num_blocks, block_size, D)
        blocks = torch.stack(blocks, dim=2)
        N = blocks.size(2)  # Number of blocks
        
        # Reshape for compression MLP: (B*H*N, block_size*D)
        blocks = blocks.view(-1, block_size * D)

        # Apply compression function φ
        compressed = self.compress_net(blocks)
        
        # Reshape back: (B, H, N, D)
        return compressed.view(B, H, N, D)
