# Transformer from Scratch

Complete implementation of transformer architecture from first principles using PyTorch.

## Overview

This repository contains a clean implementation of the transformer architecture introduced in "Attention Is All You Need" (Vaswani et al., 2017). All components built from scratch without using pre-built transformer libraries.

## Features

- **Complete Implementation**: All transformer components built from scratch
- **Clean Architecture**: Modular design with clear separation of concerns
- **Well Tested**: Comprehensive test suite
- **Documented**: Type hints and docstrings throughout
- **Educational**: Jupyter notebooks showing development process

## Installation
```bash
git clone https://github.com/antinsb2/transformer-from-scratch.git
cd transformer-from-scratch

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start
```python
from transformer import SimpleTransformer

# Create model
model = SimpleTransformer(
    vocab_size=10000,
    d_model=512,
    num_layers=6,
    num_heads=8,
    d_ff=2048
)

# Use model
import torch
tokens = torch.randint(0, 10000, (2, 10))
logits, attention_weights = model(tokens)
```

## Architecture

### Components

**Attention Mechanisms** (`attention.py`)
- Scaled dot-product attention
- Multi-head attention with configurable heads
- Support for attention masking

**Feed-Forward Networks** (`feedforward.py`)
- Position-wise feed-forward layers
- Configurable expansion factor (default 4x)

**Positional Encoding** (`encoding.py`)
- Sinusoidal positional encoding
- Token embeddings with scaling

**Transformer Blocks** (`layers.py`)
- Complete transformer block with residual connections
- Layer normalization
- Stacked encoder architecture

**Complete Model** (`model.py`)
- End-to-end transformer for next-token prediction
- Configurable architecture
- ~25M parameters (default config)

### Model Configuration
```python
SimpleTransformer(
    vocab_size=10000,    # Vocabulary size
    d_model=512,         # Model dimension
    num_layers=6,        # Number of transformer blocks
    num_heads=8,         # Attention heads per block
    d_ff=2048,          # Feed-forward hidden dimension
    max_len=5000,       # Maximum sequence length
    dropout=0.1         # Dropout rate
)
```

## Project Structure
```
transformer-from-scratch/
├── src/transformer/          # Core implementation
│   ├── attention.py         # Attention mechanisms
│   ├── feedforward.py       # Feed-forward networks
│   ├── encoding.py          # Positional encoding & embeddings
│   ├── layers.py            # Transformer blocks
│   ├── model.py             # Complete model
│   └── utils.py             # Utility functions
├── tests/                    # Test suite
│   ├── test_attention.py
│   └── test_model.py
├── examples/                 # Usage examples
│   └── train_simple.py
├── notebooks/                # Development notebooks
│   ├── 01_attention_basics.ipynb
│   ├── 02_multihead_attention.ipynb
│   ├── 03_transformer_block.ipynb
│   ├── 04_positional_encoding.ipynb
│   ├── 05_complete_transformer.ipynb
│   └── 06_test_modules.ipynb
└── README.md
```

## Running Tests
```bash
cd tests
python test_attention.py
python test_model.py
```

## Training Example
```bash
cd examples
python train_simple.py
```

Trains a small transformer (128d, 2 layers) on a toy sequence prediction task.

## Technical Details

### Attention Mechanism

Multi-head attention splits the model dimension into multiple heads, allowing the model to attend to different representation subspaces:
```
d_k = d_model / num_heads
attention(Q, K, V) = softmax(QK^T / √d_k)V
```

### Architecture Choices

- **Residual Connections**: Enable training of deep networks
- **Layer Normalization**: Stabilizes training
- **Sinusoidal Positional Encoding**: Allows model to use position information
- **Feed-forward Expansion**: 4x expansion provides capacity for complex transformations

### Parameters

Default configuration (6 layers, 512d, 8 heads):
- Total parameters: ~25M
- Attention parameters: ~12M
- Feed-forward parameters: ~12M
- Embeddings: ~5M (for 10K vocab)

## Development Process

The development process is documented in Jupyter notebooks:

1. **Basic Attention**: Understanding scaled dot-product attention
2. **Multi-Head Attention**: Parallel attention mechanisms
3. **Transformer Block**: Complete block with residuals
4. **Positional Encoding**: Adding position information
5. **Complete Model**: End-to-end training
6. **Module Testing**: Validation of production code

## Key Learnings

- **Attention is powerful but simple**: The core mechanism is straightforward matrix operations
- **Multi-head attention captures different patterns**: Each head specializes in different relationships
- **Residuals + normalization are critical**: Enable training of deep networks
- **Positional encoding is essential**: Without it, attention is permutation-invariant

## Limitations

This is an educational implementation. For production use, consider:
- Optimized attention implementations (Flash Attention)
- Mixed precision training
- Gradient checkpointing for large models
- More sophisticated tokenization

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original transformer paper
- [The Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/) - Detailed walkthrough
- [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT) - Minimal GPT implementation

## License

MIT

## Author

Antin Selvaraj - [GitHub](https://github.com/antinsb2)

---

*Built as part of a deep dive into AI systems architecture*
