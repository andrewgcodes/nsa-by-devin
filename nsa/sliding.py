import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import NSAConfig


class SlidingWindowAttention(nn.Module):
    """Handles local context through sliding window attention."""

    def __init__(self, config: NSAConfig):
        super().__init__()
        self.config = config

    def forward(
        self,
        query: torch.Tensor,   # (B, H, d_k)
        keys: torch.Tensor,    # (B, L, d_k)
        values: torch.Tensor   # (B, L, d_v)
    ) -> torch.Tensor:
        """Compute attention over sliding window."""
        window_size = self.config.window_size

        B, H, L, D = query.shape
        window_size = min(self.config.window_size, L)
        
        # Get window of recent tokens
        window_keys = keys[:, :, -window_size:]  # (B, H, w, D)
        window_values = values[:, :, -window_size:]  # (B, H, w, D)
        
        # Compute attention scores
        scores = torch.matmul(query, window_keys.transpose(-2, -1))  # (B, H, L, w)
        scores = scores / (D ** 0.5)
        
        # Apply causal mask if needed
        if self.training:
            # Create causal mask for each query position
            positions = torch.arange(L, device=scores.device).unsqueeze(-1)  # (L, 1)
            window_positions = torch.arange(window_size, device=scores.device)  # (w)
            mask = positions < window_positions  # (L, w)
            scores = scores.masked_fill(mask, float('-inf'))

        # Convert to probabilities and compute weighted sum
        attn_probs = F.softmax(scores, dim=-1)  # (B, H, L, w)
        return torch.matmul(attn_probs, window_values)  # (B, H, L, D)
