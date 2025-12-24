# Learning Log

## Dec 21 - Attention Mechanism
- Implemented scaled dot-product attention
- Created visualization of attention weights
- Built reusable PyTorch module
- Key insight: Scaling by √d_k prevents softmax saturation

## Dec 22 - Multi-Head Attention
- Implemented multi-head attention with 8 heads
- Each head learns different relationship patterns
- Added causal masking for autoregressive generation
- Compared single vs multi-head architectures
- Key insight: Multiple perspectives > single perspective

## Dec 23 - Transformer Block
- Built feed-forward network with 4x expansion (512→2048→512)
- Added layer normalization for training stability
- Implemented residual connections (prevents gradient issues)
- Stacked 6 blocks into complete transformer encoder
- Key insight: Residual + norm after each sub-layer is critical

## Dec 24 - Positional Encoding
- Implemented sinusoidal positional encoding
- Added token embeddings with sqrt(d_model) scaling
- Built complete input layer (tokens → embeddings → positions → dropout)
- Key insight: Without position info, attention can't distinguish word order
- Sin/cos allows model to learn relative positions

## Next: Complete transformer with training loop
