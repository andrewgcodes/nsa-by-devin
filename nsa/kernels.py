"""Pure PyTorch implementation of NSA attention."""

import torch
import torch.nn.functional as F

def nsa_forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Forward pass for NSA attention using PyTorch.
    
    Args:
        q: Query tensor (B, H, L, D)
        k: Key tensor (B, H, L', D)
        v: Value tensor (B, H, L', D)
        
    Returns:
        Output tensor (B, H, L, D)
    """
    # Compute attention scores
    scores = torch.matmul(q, k.transpose(-2, -1))  # (B, H, L, L')
    scores = scores / (q.size(-1) ** 0.5)
    
    # Apply attention
    attn_weights = F.softmax(scores, dim=-1)
    return torch.matmul(attn_weights, v)  # (B, H, L, D)
