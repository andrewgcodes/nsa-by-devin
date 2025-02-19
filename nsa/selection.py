import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import NSAConfig


class BlockwiseSelector(nn.Module):
    """Selects important blocks based on compressed attention scores."""

    def __init__(self, config: NSAConfig):
        super().__init__()
        self.config = config

    def compute_importance_scores(
        self,
        query: torch.Tensor,          # (B, H, L, D)
        compressed_keys: torch.Tensor  # (B, H, N, D)
    ) -> torch.Tensor:
        """Compute block importance scores using compressed attention.
        
        Following the paper's formulation:
        pᶜᵐᵖₜ = Softmax(qₜᵀK̃ᶜᵐᵖₜ)
        
        Args:
            query: Query tensor (B, H, L, D)
            compressed_keys: Compressed key tensor (B, H, N, D)
        Returns:
            Block importance scores (B, H, N)
        """
        # Use last query position qₜ
        q = query[:, :, -1:, :]  # (B, H, 1, D)
        
        # Compute attention scores: qₜᵀK̃ᶜᵐᵖₜ
        scores = torch.matmul(q, compressed_keys.transpose(-2, -1))  # (B, H, 1, N)
        scores = scores.squeeze(2)  # (B, H, N)
        scores = scores / (self.config.head_dim ** 0.5)  # Scale by √d

        # Convert to probabilities: Softmax(qₜᵀK̃ᶜᵐᵖₜ)
        probs = F.softmax(scores, dim=-1)  # (B, H, N)

        if self.config.use_gqa and probs.size(-1) > 0:
            # For GQA, aggregate scores across heads in same group
            B, H, N = probs.shape
            G = self.config.num_query_groups
            heads_per_group = H // G
            probs = probs.view(B, G, heads_per_group, N)
            probs = probs.mean(dim=2)  # (B, G, N)

        return probs

    def select_blocks(
        self,
        importance_scores: torch.Tensor,  # (B, H, N)
        keys: torch.Tensor,              # (B, H, L, D)
        values: torch.Tensor             # (B, H, L, D)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select top-n blocks based on importance scores.
        
        Following the paper's formulation:
        Iₜ = {i | rank(pˢˡᶜₜ[i]) ≤ n}
        K̃ˢˡᶜₜ = Cat{k_{il'+1:(i+1)l'} | i ∈ Iₜ}
        
        Args:
            importance_scores: Block importance scores (B, H, N)
            keys: Input keys (B, H, L, D)
            values: Input values (B, H, L, D)
        Returns:
            Selected keys and values (B, H, n*block_size, D)
        """
        B = keys.size(0)
        block_size = self.config.selection_block_size
        n = self.config.num_selected_blocks

        # Get indices of top-n blocks: Iₜ = {i | rank(pˢˡᶜₜ[i]) ≤ n}
        _, indices = torch.topk(
            importance_scores,
            k=min(n, importance_scores.size(-1)),
            dim=-1
        )  # (B, H, n)
        
        # Get shapes
        B, num_heads, _, D = keys.size()
        
        # Convert indices to block starts, ensuring we don't exceed sequence length
        seq_len = keys.size(2)
        max_blocks = (seq_len - block_size + 1) // block_size
        indices = torch.clamp(indices, 0, max_blocks - 1)
        block_starts = indices * block_size  # (B, H, n)
        
        # Create offsets for each position in block
        offsets = torch.arange(block_size, device=keys.device)
        
        # Expand indices for gathering
        gather_indices = (
            block_starts.unsqueeze(-1) +  # (B, H, n, 1)
            offsets.view(1, 1, 1, -1)     # (1, 1, 1, block_size)
        )  # (B, H, n, block_size)
        
        # Ensure indices don't exceed sequence length
        gather_indices = torch.clamp(gather_indices, 0, seq_len - 1)
        
        # Expand indices for all feature dimensions
        gather_indices = gather_indices.unsqueeze(-1)  # (B, H, n, block_size, 1)
        gather_indices = gather_indices.expand(-1, -1, -1, -1, D)
        
        # Gather blocks
        selected_keys = torch.gather(keys, 2, gather_indices.view(B, num_heads, -1, D))
        selected_values = torch.gather(values, 2, gather_indices.view(B, num_heads, -1, D))

        return selected_keys, selected_values
