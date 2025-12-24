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

notebooks/
├── 01_attention_basics.ipynb       # Scaled dot-product attention
├── 02_multihead_attention.ipynb    # Multi-head mechanism
├── 03_transformer_block.ipynb      # Complete transformer block
├── 04_positional_encoding.ipynb    # Position embeddings
└── 05_complete_transformer.ipynb   # Full model + training
```

## Key Learnings

**Attention**: Computes relationships between sequence positions
**Multi-head**: Multiple parallel attention patterns for richer representations
**Residuals + LayerNorm**: Critical for training deep networks
**Positional Encoding**: Injects sequence order information
**Training**: Standard next-token prediction with cross-entropy loss

## Status
✅ Week 1 complete (Days 1-5)
🔄 Next: Code refactoring and production modules

## Visualizations
- Attention weight heatmaps
- Multi-head pattern analysis
- Positional encoding structure
- Training loss curves

---

*Part of year-long deep dive into AI systems engineering*
