from dataclasses import dataclass


@dataclass
class NSAConfig:
    # Model dimensions
    hidden_dim: int
    num_heads: int
    head_dim: int

    # Block sizes
    compression_block_size: int = 64  # l in paper
    compression_stride: int = 32      # d in paper
    selection_block_size: int = 64    # l' in paper
    window_size: int = 256           # w in paper

    # Selection parameters
    num_selected_blocks: int = 8     # n in paper

    # Architecture
    use_gqa: bool = True            # Whether to use Grouped Query Attention
    num_query_groups: int = 4       # Number of query groups for GQA

    def __post_init__(self):
        assert self.hidden_dim % self.num_heads == 0
        assert self.compression_stride <= self.compression_block_size
        if self.use_gqa:
            assert self.num_heads % self.num_query_groups == 0
