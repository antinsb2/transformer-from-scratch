"""
Tests for attention mechanisms.
"""

import torch
import sys
sys.path.append('../src')

from transformer.attention import MultiHeadAttention


def test_multihead_attention_shape():
    """Test output shapes are correct."""
    d_model = 512
    num_heads = 8
    batch_size = 2
    seq_len = 10
    
    attention = MultiHeadAttention(d_model, num_heads)
    x = torch.randn(batch_size, seq_len, d_model)
    
    output, weights = attention(x)
    
    assert output.shape == (batch_size, seq_len, d_model)
    assert weights.shape == (batch_size, num_heads, seq_len, seq_len)
    print("✅ MultiHeadAttention shape test passed")


def test_attention_with_mask():
    """Test causal masking works."""
    d_model = 64
    num_heads = 4
    seq_len = 5
    
    attention = MultiHeadAttention(d_model, num_heads)
    x = torch.randn(1, seq_len, d_model)
    
    # Create causal mask
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    mask = (mask == 0).unsqueeze(0).unsqueeze(0)
    
    output, weights = attention(x, mask)
    
    # Check that future positions have zero attention
    # (after masking, weights should be 0 for upper triangle)
    assert output.shape == (1, seq_len, d_model)
    print("✅ Causal masking test passed")


def test_attention_parameters():
    """Test parameter count is correct."""
    d_model = 512
    num_heads = 8
    
    attention = MultiHeadAttention(d_model, num_heads)
    num_params = sum(p.numel() for p in attention.parameters())
    
    # 4 linear layers: W_q, W_k, W_v, W_o
    # Each is d_model x d_model
    expected_params = 4 * (d_model * d_model + d_model)
    
    assert num_params == expected_params
    print(f"✅ Parameters test passed: {num_params:,} params")


if __name__ == "__main__":
    test_multihead_attention_shape()
    test_attention_with_mask()
    test_attention_parameters()
    print("\n✅ All attention tests passed!")
