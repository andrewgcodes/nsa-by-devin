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
        query: torch.Tensor,   # (B, H, L, D)
        keys: torch.Tensor,    # (B, H, L, D)
        values: torch.Tensor   # (B, H, L, D)
    ) -> torch.Tensor:
        """Compute attention over sliding window.
        
        Following the paper's formulation:
        K̃ʷⁱⁿₜ = k_{t-w:t}, Ṽʷⁱⁿₜ = v_{t-w:t}
        where w is the window size
        
        Args:
            query: Query tensor (B, H, L, D)
            keys: Key tensor (B, H, L, D)
            values: Value tensor (B, H, L, D)
        Returns:
            Output tensor (B, H, L, D)
        """
        B, H, L, D = query.shape
        window_size = min(self.config.window_size, L)  # Ensure window fits sequence
        
        # Extract sliding window for each position
        # For position t, we use tokens [t-w:t] as context
        # This creates a sliding window of size w for each query position
        
        # Get window of recent tokens: K̃ʷⁱⁿₜ = k_{t-w:t}, Ṽʷⁱⁿₜ = v_{t-w:t}
        window_keys = keys[:, :, -window_size:]  # (B, H, w, D)
        window_values = values[:, :, -window_size:]  # (B, H, w, D)
        
        # Compute attention scores: qₜᵀK̃ʷⁱⁿₜ
        scores = torch.matmul(query, window_keys.transpose(-2, -1))  # (B, H, L, w)
        scores = scores / (D ** 0.5)  # Scale by √d for stable training
        
        # Apply causal masking during training to prevent attending to future tokens
        if self.training:
            # Create causal mask ensuring each position only attends to past tokens
            positions = torch.arange(L, device=scores.device).unsqueeze(-1)  # (L, 1)
            window_positions = torch.arange(window_size, device=scores.device)  # (w)
            mask = positions < window_positions  # (L, w)
            scores = scores.masked_fill(mask, float('-inf'))  # Mask out future tokens

        # Convert to probabilities: Softmax(qₜᵀK̃ʷⁱⁿₜ/√d)
        attn_probs = F.softmax(scores, dim=-1)  # (B, H, L, w)
        
        # Compute weighted sum: Attn(qₜ, K̃ʷⁱⁿₜ, Ṽʷⁱⁿₜ)
        return torch.matmul(attn_probs, window_values)  # (B, H, L, D)
