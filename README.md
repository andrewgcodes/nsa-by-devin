# NSA (Natively trainable Sparse Attention)

Implementation of NSA attention mechanism based on the paper "NSA: Natively trainable Sparse Attention for Efficient Long-Context Modeling".

Key components:
1. Token Compression - Aggregates sequential blocks into compressed representations
2. Token Selection - Blockwise selection of important tokens
3. Sliding Window - Dedicated branch for local context
4. Hardware-optimized kernels for efficient computation

Requirements:
- PyTorch for tensor operations
- Triton for custom CUDA kernels
- CUDA support for hardware acceleration
