"""
Tests for complete transformer model.
"""

import torch
import sys
sys.path.append('../src')

from transformer.model import SimpleTransformer


def test_model_forward():
    """Test model forward pass."""
    vocab_size = 1000
    d_model = 128
    num_layers = 2
    num_heads = 4
    d_ff = 512
    
    model = SimpleTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff
    )
    
    batch_size = 2
    seq_len = 10
    tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    logits, attention_weights = model(tokens)
    
    assert logits.shape == (batch_size, seq_len, vocab_size)
    assert len(attention_weights) == num_layers
    print("✅ Model forward pass test passed")


def test_model_training_step():
    """Test that model can train (backward pass works)."""
    vocab_size = 100
    model = SimpleTransformer(
        vocab_size=vocab_size,
        d_model=64,
        num_layers=2,
        num_heads=2,
        d_ff=128
    )
    
    # Create dummy data
    tokens = torch.randint(0, vocab_size, (2, 10))
    targets = torch.randint(0, vocab_size, (2, 10))
    
    # Forward pass
    logits, _ = model(tokens)
    
    # Compute loss
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, vocab_size),
        targets.view(-1)
    )
    
    # Backward pass
    loss.backward()
    
    # Check gradients exist
    has_grad = any(p.grad is not None for p in model.parameters())
    assert has_grad
    print("✅ Training step test passed")


def test_model_parameter_count():
    """Test parameter count makes sense."""
    model = SimpleTransformer(
        vocab_size=10000,
        d_model=512,
        num_layers=6,
        num_heads=8,
        d_ff=2048
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    
    # Should be in the ~25M range for this config
    assert 20_000_000 < num_params < 30_000_000
    print(f"✅ Parameter count test passed: {num_params:,} params")


if __name__ == "__main__":
    test_model_forward()
    test_model_training_step()
    test_model_parameter_count()
    print("\n✅ All model tests passed!")
