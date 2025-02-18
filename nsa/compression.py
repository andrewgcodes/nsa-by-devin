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

        # Learnable compression MLP
        self.compress_net = nn.Sequential(
            nn.Linear(
                config.head_dim * config.compression_block_size,
                config.head_dim * 4
            ),
            nn.GELU(),
            nn.Linear(config.head_dim * 4, config.head_dim)
        )

        # Learnable position encoding for intra-block positions
        self.pos_embedding = nn.Parameter(
            torch.randn(1, config.compression_block_size, config.head_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, num_heads, seq_len, head_dim)
        Returns:
            compressed: (batch_size, num_heads, compressed_len, head_dim)
        """
        B, H, L, D = x.shape
        block_size = self.config.compression_block_size
        stride = self.config.compression_stride

        # Add positional encoding
        blocks = []
        for i in range(0, L - block_size + 1, stride):
            # Extract block and add positional encoding
            # Reshape pos_embedding to match dimensions
            pos_emb = self.pos_embedding.view(1, 1, block_size, D)
            block = x[:, :, i:i+block_size, :] + pos_emb
            blocks.append(block)

        if not blocks:
            return x.new_zeros((B, H, 0, D))

        # Stack blocks: (B, H, num_blocks, block_size, D)
        blocks = torch.stack(blocks, dim=2)
        N = blocks.size(2)  # Number of blocks
        
        # Reshape for compression: (B*H*N, block_size*D)
        blocks = blocks.view(-1, block_size * D)

        # Compress blocks
        compressed = self.compress_net(blocks)
        
        # Reshape back: (B, H, N, D)
        return compressed.view(B, H, N, D)
