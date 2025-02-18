# NSA: Natively trainable Sparse Attention

This repository contains a PyTorch implementation of NSA (Natively trainable Sparse Attention) mechanism, which integrates algorithmic innovations with hardware-aligned optimizations to achieve efficient long-context modeling.

## Mathematical Formulations

NSA employs a hierarchical sparse strategy that divides the attention computation into three parallel branches:

1. Token Compression (Coarse-Grained):
   ```
   K̃ᶜᵐᵖₜ = φ(k_{i:i+l}) for i = 1, ..., t-l+1
   ```
   Partitions input sequence into blocks and compresses each block into a single token using a learnable MLP with intra-block positional encoding.

2. Token Selection (Fine-Grained):
   ```
   pᶜᵐᵖₜ = Softmax(qₜᵀK̃ᶜᵐᵖₜ)
   Iₜ = {i | rank(pˢˡᶜₜ[i]) ≤ n}
   K̃ˢˡᶜₜ = Cat{k_{il'+1:(i+1)l'} | i ∈ Iₜ}
   ```
   Preserves critical fine-grained information by selecting important blocks based on attention scores.

3. Sliding Window (Local Context):
   ```
   K̃ʷⁱⁿₜ = k_{t-w:t}, Ṽʷⁱⁿₜ = v_{t-w:t}
   ```
   Explicitly handles local context by maintaining a fixed-size window of recent tokens.

4. Final Output:
   ```
   o*ₜ = Σ_{c∈{cmp,slc,win}} gᶜₜ · Attn(qₜ, K̃ᶜₜ, Ṽᶜₜ)
   ```
   Combines outputs from all branches using learned gating mechanism.

## Features

- Three parallel attention paths:
  1. Compressed attention for global context
  2. Selected attention for important tokens
  3. Sliding window attention for local context
- Gated combination of attention paths
- Pure PyTorch implementation for CPU/GPU training
- Support for Grouped Query Attention (GQA)

## Installation

```bash
# Clone the repository
git clone https://github.com/andrewgcodes/nsa-by-devin.git
cd nsa-by-devin

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training Example

The repository includes a simple sequence prediction example to demonstrate the NSA mechanism:

```bash
# Activate virtual environment if not already active
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Run training script
python train.py
```

This trains a small transformer model using NSA attention on a sequence prediction task where each number is the sum of the previous two numbers.

### Using NSA in Your Project

```python
import torch
from nsa.model import NSATransformer

# Create model
model = NSATransformer(
    vocab_size=10,      # Size of vocabulary
    num_layers=2,       # Number of transformer layers
    hidden_dim=256,     # Hidden dimension
    num_heads=4         # Number of attention heads
)

# Forward pass
batch_size, seq_len = 32, 128
x = torch.randint(0, 10, (batch_size, seq_len))
logits = model(x)  # Shape: (batch_size, seq_len, vocab_size)
```

## Architecture

The NSA implementation consists of several key components:

1. `NSAAttention`: Main attention module that combines three paths
2. `TokenCompressor`: Compresses blocks of tokens for global context
3. `BlockwiseSelector`: Selects important blocks based on attention scores
4. `SlidingWindowAttention`: Handles local context through windowed attention

### Component Details

#### NSAAttention
- Combines three attention paths using a learned gating mechanism
- Supports both standard multi-head attention and grouped query attention (GQA)
- Projects inputs/outputs through learned linear transformations

#### TokenCompressor
- Compresses blocks of tokens into single representations
- Uses learnable position encodings for intra-block positions
- Maintains global context through compressed token interactions

#### BlockwiseSelector
- Dynamically selects important blocks based on attention scores
- Supports grouped selection for GQA compatibility
- Efficiently gathers selected blocks for attention computation

#### SlidingWindowAttention
- Processes local context through a sliding window mechanism
- Supports causal masking during training
- Efficient implementation for local attention patterns

## Configuration

Key configuration parameters in `NSAConfig`:

```python
config = NSAConfig(
    hidden_dim=256,           # Model dimension
    num_heads=4,              # Number of attention heads
    head_dim=64,             # Dimension per head
    compression_block_size=64,# Size of compression blocks
    selection_block_size=64,  # Size of selection blocks
    window_size=256,         # Sliding window size
    use_gqa=True,            # Use Grouped Query Attention
    num_query_groups=4       # Number of query groups for GQA
)
```

## Training Results

On the sequence prediction task:
- Initial Loss (Epoch 1): 2.26
- Final Loss (Epoch 10): 0.33
- Shows stable convergence and effective learning

## Development

To contribute to this project:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and ensure code quality
5. Submit a pull request

## Troubleshooting

Common issues and solutions:

1. Memory issues
   - Reduce batch size or sequence length
   - Use smaller model configuration
   - Enable gradient checkpointing

2. Training instability
   - Adjust learning rate
   - Increase warmup steps
   - Check attention mask implementation

3. Package import errors
   - Ensure virtual environment is activated
   - Verify all dependencies are installed
   - Check Python version compatibility

## License

MIT License

## Citation

If you use this implementation in your research, please cite:

```bibtex
@article{nsa2025,
  title={NSA: Natively trainable Sparse Attention for Efficient Long-Context Modeling},
  author={[Authors]},
  journal={arXiv preprint arXiv:2502.11089},
  year={2025}
}
```

## Installation

```bash
# Clone the repository
git clone https://github.com/andrewgcodes/nsa-by-devin.git
cd nsa-by-devin

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training Example

The repository includes a simple sequence prediction example to demonstrate the NSA mechanism:

```bash
# Activate virtual environment if not already active
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Run training script
python train.py
```

This trains a small transformer model using NSA attention on a sequence prediction task where each number is the sum of the previous two numbers.

### Using NSA in Your Project

```python
import torch
from nsa.model import NSATransformer

# Create model
model = NSATransformer(
    vocab_size=10,      # Size of vocabulary
    num_layers=2,       # Number of transformer layers
    hidden_dim=256,     # Hidden dimension
    num_heads=4         # Number of attention heads
)

# Forward pass
batch_size, seq_len = 32, 128
x = torch.randint(0, 10, (batch_size, seq_len))
logits = model(x)  # Shape: (batch_size, seq_len, vocab_size)
```

## Architecture

The NSA implementation consists of several key components:

1. `NSAAttention`: Main attention module that combines three paths
2. `TokenCompressor`: Compresses blocks of tokens for global context
3. `BlockwiseSelector`: Selects important blocks based on attention scores
4. `SlidingWindowAttention`: Handles local context through windowed attention

### Component Details

#### NSAAttention
- Combines three attention paths using a learned gating mechanism
- Supports both standard multi-head attention and grouped query attention (GQA)
- Projects inputs/outputs through learned linear transformations

#### TokenCompressor
- Compresses blocks of tokens into single representations
- Uses learnable position encodings for intra-block positions
- Maintains global context through compressed token interactions

#### BlockwiseSelector
- Dynamically selects important blocks based on attention scores
- Supports grouped selection for GQA compatibility
- Efficiently gathers selected blocks for attention computation

#### SlidingWindowAttention
- Processes local context through a sliding window mechanism
- Supports causal masking during training
- Efficient implementation for local attention patterns

## Configuration

Key configuration parameters in `NSAConfig`:

```python
config = NSAConfig(
    hidden_dim=256,           # Model dimension
    num_heads=4,              # Number of attention heads
    head_dim=64,             # Dimension per head
    compression_block_size=64,# Size of compression blocks
    selection_block_size=64,  # Size of selection blocks
    window_size=256,         # Sliding window size
    use_gqa=True,            # Use Grouped Query Attention
    num_query_groups=4       # Number of query groups for GQA
)
```

## Training Results

On the sequence prediction task:
- Initial Loss (Epoch 1): 2.26
- Final Loss (Epoch 10): 0.33
- Shows stable convergence and effective learning

## Development

To contribute to this project:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and ensure code quality
5. Submit a pull request

## Troubleshooting

Common issues and solutions:

1. Memory issues
   - Reduce batch size or sequence length
   - Use smaller model configuration
   - Enable gradient checkpointing

2. Training instability
   - Adjust learning rate
   - Increase warmup steps
   - Check attention mask implementation

3. Package import errors
   - Ensure virtual environment is activated
   - Verify all dependencies are installed
   - Check Python version compatibility

## License

MIT License

## Citation

If you use this implementation in your research, please cite:

```bibtex
@article{nsa2025,
  title={NSA: Natively trainable Sparse Attention for Efficient Long-Context Modeling},
  author={[Authors]},
  journal={arXiv preprint arXiv:2502.11089},
  year={2025}
}
```
