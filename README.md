# Transformer from Scratch

Complete implementation of transformer architecture from first principles.

## What's Built

### Core Components
- **Attention Mechanisms**
  - Scaled dot-product attention
  - Multi-head attention (8 heads)
  - Causal masking for autoregressive generation

- **Transformer Architecture**
  - Feed-forward networks (4x expansion: 512→2048→512)
  - Layer normalization and residual connections
  - Positional encoding (sinusoidal)
  - Token embeddings
  - Complete encoder stack (6 layers)

- **Training Pipeline**
  - End-to-end training loop
  - Loss tracking and visualization
  - Next-token prediction capability

### Model Specifications
- Architecture: GPT-style decoder-only transformer
- Parameters: ~25M (full model) / ~250K (toy model)
- Default config: 512d model, 8 heads, 2048d FFN, 6 layers

## Implementation

All components built from scratch using PyTorch:
- No pre-built transformer libraries
- Educational focus with detailed comments
- Visualizations for understanding attention patterns

## Structure
```
src/transformer/          # Production modules
├── attention.py         # Multi-head attention
├── feedforward.py       # Position-wise FFN
├── encoding.py          # Positional encoding & embeddings
├── layers.py            # Transformer blocks
└── model.py             # Complete model

notebooks/               # Exploratory implementations
├── 01_attention_basics.ipynb
├── 02_multihead_attention.ipynb
├── 03_transformer_block.ipynb
├── 04_positional_encoding.ipynb
├── 05_complete_transformer.ipynb
└── 06_test_modules.ipynb
```

**Attention**: Computes relationships between sequence positions
**Multi-head**: Multiple parallel attention patterns for richer representations
**Residuals + LayerNorm**: Critical for training deep networks
**Positional Encoding**: Injects sequence order information
**Training**: Standard next-token prediction with cross-entropy loss



