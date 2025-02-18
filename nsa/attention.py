"""Main NSA attention implementation combining all components."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import NSAConfig
from .compression import TokenCompressor
from .selection import BlockwiseSelector
from .sliding import SlidingWindowAttention
from .kernels import nsa_forward


class NSAAttention(nn.Module):
    """Native Sparse Attention implementation."""
    def __init__(self, config: NSAConfig):
        super().__init__()
        self.config = config

        # Components
        self.compressor = TokenCompressor(config)
        self.selector = BlockwiseSelector(config)
        self.sliding = SlidingWindowAttention(config)

        # Gating mechanism
        self.gate_net = nn.Linear(config.hidden_dim, 3)  # 3 paths

        # Projections
        self.q_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.k_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.v_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.o_proj = nn.Linear(config.hidden_dim, config.hidden_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,  # (B, L, H)
        attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        B, L, H = hidden_states.shape

        # Project and reshape inputs
        head_dim = self.config.hidden_dim // self.config.num_heads
        q = self.q_proj(hidden_states).view(B, L, self.config.num_heads, head_dim)
        k = self.k_proj(hidden_states).view(B, L, self.config.num_heads, head_dim)
        v = self.v_proj(hidden_states).view(B, L, self.config.num_heads, head_dim)
        
        # Transpose to (B, H, L, D)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compute gates following paper's formulation:
        # o*ₜ = Σ_{c∈{cmp,slc,win}} gᶜₜ · Attn(qₜ, K̃ᶜₜ, Ṽᶜₜ)
        # where gᶜₜ ∈ [0,1] via sigmoid activation
        gates = F.sigmoid(self.gate_net(hidden_states))  # (B, L, 3)

        # 1. Compressed attention (K̃ᶜᵐᵖₜ = φ(k_{i:i+l}))
        compressed_k = self.compressor(k)
        compressed_v = self.compressor(v)
        compressed_out = nsa_forward(q, compressed_k, compressed_v)

        # 2. Selected attention (pᶜᵐᵖₜ = Softmax(qₜᵀK̃ᶜᵐᵖₜ))
        importance_scores = self.selector.compute_importance_scores(
            q, compressed_k
        )
        selected_k, selected_v = self.selector.select_blocks(
            importance_scores, k, v
        )
        selected_out = nsa_forward(q, selected_k, selected_v)

        # 3. Sliding window attention (K̃ʷⁱⁿₜ = k_{t-w:t})
        sliding_out = self.sliding(q, k, v)

        # Apply gated combination
        gates = gates.unsqueeze(1)  # (B, 1, L, 3)
        
        # Combine outputs with gated weighting
        out = (
            gates[..., 0:1] * compressed_out +  # gᶜᵐᵖₜ · Attn(qₜ, K̃ᶜᵐᵖₜ, Ṽᶜᵐᵖₜ)
            gates[..., 1:2] * selected_out +    # gˢˡᶜₜ · Attn(qₜ, K̃ˢˡᶜₜ, Ṽˢˡᶜₜ)
            gates[..., 2:3] * sliding_out       # gʷⁱⁿₜ · Attn(qₜ, K̃ʷⁱⁿₜ, Ṽʷⁱⁿₜ)
        )

        # Project output
        out = out.transpose(1, 2).contiguous().view(B, L, H)
        return self.o_proj(out)
